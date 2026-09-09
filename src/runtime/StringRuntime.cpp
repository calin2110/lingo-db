#include "lingodb/runtime/StringRuntime.h"
#include "arrow/util/formatting.h"
#include "arrow/util/value_parsing.h"
#include "lingodb/runtime/helpers.h"

#include <regex>

#include <arrow/type.h>
#include <arrow/util/decimal.h>

#ifdef __SSE4_2__
#include <nmmintrin.h>
#include <pmmintrin.h>
#endif
//taken from NoisePage
// src: https://github.com/cmu-db/noisepage/blob/c2635d3360dd24a9f7a094b4b8bcd131d99f2d4b/src/execution/sql/operators/like_operators.cpp
// (MIT License, Copyright (c) 2018 CMU Database Group)
#define NextByte(p, plen) ((p)++, (plen)--)

namespace {

// can be combined with the NextChar in the StringRuntime, but would need iterativeLike to be rewritten
void nextChar(const char*& p, std::size_t& plen) {
   // handle first byte and continuations
   do {
      p++;
      plen--;
   } while (plen > 0 &&
            (static_cast<uint8_t>(*p) >> 6) == 2);
}

bool iterativeLike(const char* str, size_t strLen, const char* pattern, size_t patternLen, char escape) {
   const char *s = str, *p = pattern;
   std::size_t slen = strLen, plen = patternLen;

   for (; plen > 0 && slen > 0; nextChar(p, plen)) {
      if (*p == escape) {
         // Next pattern character must match exactly, whatever it is
         nextChar(p, plen);

         if (plen == 0 || *p != *s) {
            return false;
         }

         nextChar(s, slen);
      } else if (*p == '%') {
         // Any sequence of '%' wildcards can essentially be replaced by one '%'. Similarly, any
         // sequence of N '_'s will blindly consume N characters from the input string. Process the
         // pattern until we reach a non-wildcard character.
         nextChar(p, plen);
         while (plen > 0) {
            if (*p == '%') {
               nextChar(p, plen);
            } else if (*p == '_') {
               if (slen == 0) {
                  return false;
               }
               nextChar(s, slen);
               nextChar(p, plen);
            } else {
               break;
            }
         }

         // If we've reached the end of the pattern, the tail of the input string is accepted.
         if (plen == 0) {
            return true;
         }

         if (*p == escape) {
            nextChar(p, plen);
            if (plen == 0) {
               return false;
            }
         }

         while (slen > 0) {
            if (iterativeLike(s, slen, p, plen, escape)) {
               return true;
            }
            nextChar(s, slen);
         }
         // No match
         return false;
      } else if (*p == '_') {
         // '_' wildcard matches a single character in the input
         nextChar(s, slen);
      } else if (*p == *s) {
         // Exact character match
         nextChar(s, slen);
      } else {
         // Unmatched!
         return false;
      }
   }
   while (plen > 0 && *p == '%') {
      nextChar(p, plen);
   }
   return slen == 0 && plen == 0;
}

const uint8_t* twoWaySearch(const uint8_t* haystack, int32_t searchLimit, const uint8_t* needle, int32_t needleLen, int32_t period, int32_t maxSuffix) {
      // because searchLimit is the first index for which we can no longer match
   // we find the first instance of the first two matching characters, but up to searchLimit, exclusive
   const uint8_t* haystackStart = haystack;
   const uint8_t* firstPositionStartEnd = haystack + searchLimit;
   while (true) {
      haystack = static_cast<const uint8_t*>(memchr(haystack, needle[0], firstPositionStartEnd - haystack));
      if (!haystack)
         return nullptr;
      if (haystack[1] == needle[1])
         break;
      if (++haystack == firstPositionStartEnd)
         return nullptr;
   }
   // move the haystack up to where we have found the start of the index
   // move the haystackLength to agree with the previous haystack move (subtract the value of index)
   // subtract 1 to make comparison inclusive now (search is up to hayStackLen, inclusive)
   int32_t haystackLen = searchLimit - (haystack - haystackStart) - 1;

   // equal distinguishes in which algorithm from the paper we are
   // equal set to true => POSITIONS (Figure 8)
   // equal set to false => POSITIONS-BIS (Figure 20)
   bool equal = period & 1;
   period >>= 1;

   int32_t pos, lastPtr = -1;
   int32_t resetPtr = needleLen - period - 1;
   int32_t offset = 0;

   if (!equal) {
      // http://monge.univ-mlv.fr/~mac/Articles-PDF/CP-1991-jacm.pdf, algorithm POSITIONS-BIS (Figure 20)
      // period corresponds to q
      // maxSuffix corresponds to l
      // haystackLen corresponds to the last position (inclusive) where matching all the patterns may start
      // offset corresponds to pos
      // pos corresponds to i and j
      // pre-requisites for the paper code: `period` is at most the period of the needle and `maxSuffix` is a critical position satisfying strictly smaller than the period of the needle
      while (offset <= haystackLen) {
         pos = maxSuffix + 1;
         while (pos < needleLen && needle[pos] == haystack[pos + offset]) {
            pos++;
         }
         if (pos < needleLen) {
            offset += pos - maxSuffix;
         } else {
            pos = maxSuffix;
            while (pos >= 0 && needle[pos] == haystack[pos + offset]) {
               pos--;
            }
            if (pos >= 0) {
               offset += period;
            } else {
               return haystack + offset;
            }
         }
      }
   } else {
      // http://monge.univ-mlv.fr/~mac/Articles-PDF/CP-1991-jacm.pdf, algorithm POSITIONS (Figure 8)
      // lastPtr represents the value of s from the code
      // resetPtr corresponds to the value in the right-hand side of Line 9, accounted for 0-indexing rather than 0 indexing
      // period corresponds to p
      // maxSuffix corresponds to l
      // haystackLen corresponds to the last position (inclusive) where matching all the patterns may start
      // offset corresponds to pos
      // pos corresponds to i and j
      // pre-requisites for the paper code: `period` is a period of the needle and `maxSuffix` is a critical position satisfying `maxSuffix < period`
      // but then, `pos - maxSuffix > pos - period >= lastPtr - period + 1`, so we can skip the max on line 4 entirely
      while (offset <= haystackLen) {
         pos = std::max(maxSuffix, lastPtr) + 1;
         while (pos < needleLen && needle[pos] == haystack[pos + offset]) {
            pos++;
         }
         if (pos < needleLen) {
            lastPtr = -1;
            offset += pos - maxSuffix;
         } else {
            // match.
            pos = maxSuffix;
            while (pos > lastPtr && needle[pos] == haystack[pos + offset]) {
               pos--;
            }
            if (pos > lastPtr) {
               lastPtr = resetPtr;
               offset += period;
            } else {
               return haystack + offset;
            }
         }
      }
   }
   return nullptr;
}

#ifdef __SSE4_2__
const uint8_t* simdSearch(const uint8_t* haystack, const uint8_t* haystackEnd, const uint8_t* needle, int32_t needleLen) {
   __m128i needleSIMD;
   {
      alignas(16) uint8_t bytes[16] = {0};
      memcpy(bytes, needle, needleLen);
      needleSIMD = _mm_load_si128(reinterpret_cast<const __m128i*>(bytes));
   }

   int32_t lastFullMatchIndex = 16 - needleLen;
   while (haystack + 16 <= haystackEnd) {
      __m128i haystackSIMD = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(haystack));
      int match = _mm_cmpistri(needleSIMD, haystackSIMD, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED);
      if (match <= lastFullMatchIndex) {
         return haystack + match;
      }
      haystack += match;
   }
   haystack = haystackEnd - 16;
   __m128i haystackSIMD = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(haystack));
   int match = _mm_cmpistri(needleSIMD, haystackSIMD, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED);
   return match <= lastFullMatchIndex ? haystack + match : nullptr;
}
#endif

const uint8_t* hybridSearch(const uint8_t* haystack, int32_t searchLimit, const uint8_t* needle, int32_t needleLen, int32_t period, int32_t maxSuffix) {
   if (!needleLen)
      return haystack;

   if (needleLen == 1)
      return static_cast<const uint8_t*>(memchr(haystack, needle[0], searchLimit));

   #ifdef __SSE4_2__
   if (needleLen <= 12) {
      const uint8_t* haystackEnd = haystack + searchLimit + needleLen - 1;
      if (haystackEnd - haystack >= 16)
         return simdSearch(haystack, haystackEnd, needle, needleLen);
   }
   #endif
   return twoWaySearch(haystack, searchLimit, needle, needleLen, period, maxSuffix);
}

inline constexpr uint8_t kUtf8Len[32] = {
   1,1,1,1,1,1,1,1,  1,1,1,1,1,1,1,1,   // 0x00-0x7F
   1,1,1,1,1,1,1,1,                     // 0x80-0xBF continuation
   2,2,2,2,                             // 0xC0-0xDF
   3,3,                                 // 0xE0-0xEF
   4,                                   // 0xF0-0xF7
   1                                    // 0xF8-0xFF invalid
};

constexpr uint8_t getLengthOfUTF8Sequence(uint8_t firstByte) noexcept {
   return kUtf8Len[firstByte >> 3];
}

constexpr uint8_t getMaxUtf8Length() noexcept {
   uint8_t maxLen = 0;
   for (uint8_t value: kUtf8Len) {
      maxLen = value > maxLen ? value : maxLen;
   }
   return maxLen;
}

const uint8_t* movePointerToCharacterStart(const uint8_t* reader) {
   while (((*reader) & 0xC0) == 0x80) {
      --reader;
   }
   return reader;
}

const uint8_t* likeProgramWithUnderscoresStep(const uint8_t* haystack, const uint8_t* haystackEnd, const uint8_t* pattern, const uint8_t* patternEnd, int32_t bufferLen, int32_t sums, int32_t period, int32_t maxSuffix) {
   while (true) {
      checkAgain:
      if ((haystackEnd - haystack) < sums)
         return nullptr;
      int32_t searchLimit = (haystackEnd - haystack)  - sums + 1;
      haystack = hybridSearch(haystack, searchLimit, pattern, bufferLen, period, maxSuffix);
      if (!haystack || ((haystackEnd - haystack) < sums))
         return nullptr;

      const uint8_t* reader = haystack + bufferLen;
      int32_t numSteps = 0;
      for (const uint8_t* iter = pattern + bufferLen; iter != patternEnd; ++iter) {
         if (reader == haystackEnd)
            return nullptr;
         uint8_t c = *iter;
         // in case of an actual underscore, move the pointer after the current character
         if (c == UNDERSCORE_REPLACEMENT) {
            reader += getLengthOfUTF8Sequence(*reader);
            numSteps += getMaxUtf8Length();
            continue;
         }

         // in case of a real character, if matching, move forward
         // otherwise, find the first appearance of this mismatched character and align it pessimistically with our current pattern
         // by pessimistically, I mean that we assume all single underscore characters have their maximum lengths of 6 (this is used by numSteps)
         // trivially, we aim to move only forward
         if (*reader != c) {
            const uint8_t* next = static_cast<const uint8_t*>(memchr(reader, c, haystackEnd - reader));
            if (!next)
               return nullptr;
            const uint8_t* candidate = next - numSteps - bufferLen;
            haystack = std::max(haystack + 1, candidate);
            goto checkAgain;;
         }
         ++reader;
         ++numSteps;
      }
      return reader;
   }
   return nullptr;
}
} // namespace
//end taken from noisepage

namespace {

size_t charIndexToByteIndex(lingodb::runtime::VarLen32& str, size_t charIndex, size_t knownByteIndex = 0, size_t knownCharIndex = 0) {
   /*
    * considered the following: extract first bits, check number of values needed to jump from a memory table, do jump
    * decided against it: the following implementation may be easier to vectorize
    */

   /*
    * knownByteIndex and knownCharIndex: already known mappings from previous runs
    * charIndex < len(str) length being in the utf-8 sense
    * knownByteIndex should map to the first byte of the knownCharIndex
    */

   char* data = str.data();
   uint32_t byteLen = str.getLen();

   for (; knownByteIndex < byteLen; knownByteIndex++) {
      const unsigned char c = data[knownByteIndex];
      uint8_t shifted = c >> 6;

      // if not a continuation
      if (shifted != 2) {
         if (knownCharIndex == charIndex) {
            return knownByteIndex;
         }
         knownCharIndex++;
      }
   }

   return knownByteIndex; // returns the byteLength;
}
} // namespace

bool lingodb::runtime::StringRuntime::like(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   return iterativeLike((str1).data(), (str1).getLen(), (str2).data(), (str2).getLen(), '\\');
}
bool lingodb::runtime::StringRuntime::endsWith(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   if (str1.getLen() < str2.getLen()) return false;
   return std::string_view(str1.data(), str1.getLen()).ends_with(std::string_view(str2.data(), str2.getLen()));
}
bool lingodb::runtime::StringRuntime::startsWith(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   if (str1.getLen() < str2.getLen()) return false;
   return std::string_view(str1.data(), str1.getLen()).starts_with(std::string_view(str2.data(), str2.getLen()));
}

//taken from gandiva
//source https://github.com/apache/arrow/blob/41d115071587d68891b219cc137551d3ea9a568b/cpp/src/gandiva/gdv_function_stubs.cc
//Apache-2.0 License
#define CAST_NUMERIC_FROM_STRING(OUT_TYPE, ARROW_TYPE, TYPE_NAME)                                                                                 \
   OUT_TYPE lingodb::runtime::StringRuntime::to##TYPE_NAME(lingodb::runtime::VarLen32 str) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      char* data = (str).data();                                                                                                                  \
      int32_t len = (str).getLen();                                                                                                               \
      OUT_TYPE val = 0;                                                                                                                           \
      /* trim leading and trailing spaces */                                                                                                      \
      int32_t trimmed_len;                                                                                                                        \
      int32_t start = 0, end = len - 1;                                                                                                           \
      while (start <= end && data[start] == ' ') {                                                                                                \
         ++start;                                                                                                                                 \
      }                                                                                                                                           \
      while (end >= start && data[end] == ' ') {                                                                                                  \
         --end;                                                                                                                                   \
      }                                                                                                                                           \
      trimmed_len = end - start + 1;                                                                                                              \
      const char* trimmed_data = data + start;                                                                                                    \
      if (!arrow::internal::ParseValue<ARROW_TYPE>(trimmed_data, trimmed_len, &val)) {                                                            \
         std::string err =                                                                                                                        \
            "Failed to cast the string " + std::string(data, len) + " to " #OUT_TYPE;                                                             \
         /*gdv_fn_context_set_error_msg(context, err.c_str());*/                                                                                  \
      }                                                                                                                                           \
      return val;                                                                                                                                 \
   }

//CAST_NUMERIC_FROM_STRING(int64_t, arrow::Int64Type, Int)
int64_t lingodb::runtime::StringRuntime::toInt(lingodb::runtime::VarLen32 str) {
   std::string::size_type sz = 0;
   auto res = std::stoll(str.str(), &sz);
   if (sz != str.getLen()) {
      throw std::runtime_error("string also contained non-numeric characters");
   }
   return res;
}

CAST_NUMERIC_FROM_STRING(float, arrow::FloatType, Float32)
CAST_NUMERIC_FROM_STRING(double, arrow::DoubleType, Float64)
//end taken from gandiva

__int128 lingodb::runtime::StringRuntime::toDecimal(lingodb::runtime::VarLen32 string, int32_t reqScale) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   int32_t precision;
   int32_t scale;
   arrow::Decimal128 decimalrep;
   if (!arrow::Decimal128::FromString(string.str(), &decimalrep, &precision, &scale).ok()) {
      throw std::runtime_error("could not cast decimal");
   }
   auto x = decimalrep.Rescale(scale, reqScale);
   decimalrep = x.ValueUnsafe();
   __int128 res = decimalrep.high_bits();
   res <<= 64;
   res |= decimalrep.low_bits();
   return res;
}
#define CAST_NUMERIC_TO_STRING(IN_TYPE, ARROW_TYPE, TYPE_NAME)                                                                                       \
   lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::from##TYPE_NAME(IN_TYPE value) { /* NOLINT (clang-diagnostic-return-type-c-linkage)*/ \
      arrow::internal::StringFormatter<ARROW_TYPE> formatter;                                                                                        \
      VarLen32 res;                                                                                                                                  \
      arrow::Status status = formatter(value, [&](std::string_view v) {                                                                              \
         res = VarLen32::fromString(v, StorageClass::REFCOUNTED);                                                                                    \
         return arrow::Status::OK();                                                                                                                 \
      });                                                                                                                                            \
      return res;                                                                                                                                    \
   }

CAST_NUMERIC_TO_STRING(int64_t, arrow::Int64Type, Int)
CAST_NUMERIC_TO_STRING(float, arrow::FloatType, Float32)
CAST_NUMERIC_TO_STRING(double, arrow::DoubleType, Float64)
CAST_NUMERIC_TO_STRING(bool, arrow::BooleanType, Bool)

lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::fromDecimal(__int128 val, int32_t scale) { // NOLINT (clang-diagnostic-return-type-c-linkage)

   arrow::Decimal128 decimalrep(arrow::BasicDecimal128(val >> 64, val));
   std::string str = decimalrep.ToString(scale);
   return lingodb::runtime::VarLen32::fromString(str, StorageClass::REFCOUNTED);
}

lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::fromChar(uint32_t val) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   char data[4];
   memcpy(data, &val, 4);
   size_t len;
   if ((val & (1 << 7)) == 0) {
      len = 1;
   } else if ((val & (1 << 5)) == 0) {
      len = 2;
   } else if ((val & (1 << 4)) == 0) {
      len = 3;
   } else {
      len = 4;
   }
   return lingodb::runtime::VarLen32(reinterpret_cast<uint8_t*>(data), len, StorageClass::REFCOUNTED);
}

#define STR_CMP(NAME, OP)                                                                                                  \
   bool lingodb::runtime::StringRuntime::compare##NAME(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) { \
      return std::string_view(str1.data(), str1.getLen()) OP std::string_view(str2.data(), str2.getLen());                 \
   }

STR_CMP(Lt, <)
STR_CMP(Lte, <=)
STR_CMP(Gt, >)
STR_CMP(Gte, >=)

bool lingodb::runtime::StringRuntime::compareEq(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   assert(str1.getLen() == str2.getLen() && "String length equality must be checked before calling compareEq");
   return std::string_view(str1.data(), str1.getLen()) == std::string_view(str2.data(), str2.getLen());
}
bool lingodb::runtime::StringRuntime::compareNEq(lingodb::runtime::VarLen32 str1, lingodb::runtime::VarLen32 str2) {
   if (str1.getLen() != str2.getLen()) return true;
   return std::string_view(str1.data(), str1.getLen()) != std::string_view(str2.data(), str2.getLen());
}

EXPORT char* rt_varlen_to_ref(lingodb::runtime::VarLen32* varlen) { // NOLINT (clang-diagnostic-return-type-c-linkage)
   return varlen->data();
}

size_t lingodb::runtime::StringRuntime::nextChar(lingodb::runtime::VarLen32 str, size_t position) {
   // handle first byte and continuations
   char* data = str.data();
   uint32_t byteLen = str.getLen();

   do {
      position++;
   } while (position < byteLen &&
            (static_cast<uint8_t>(data[position]) >> 6) == 2);

   return position;
}

int64_t lingodb::runtime::StringRuntime::len(VarLen32 str) {
   char* data = str.data();
   uint32_t byteLen = str.getLen();

   // start with byteLen, subtract at every continuation (starting with b10xxxxxx)
   uint32_t charLen = byteLen;

   for (uint32_t i = 0; i < byteLen; i++) {
      const unsigned char c = data[i];
      uint8_t shifted = c >> 6;
      charLen -= (shifted == 2);
   }

   return charLen;
}

lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::substr(lingodb::runtime::VarLen32 str, int64_t from, int64_t len) { // NOLINT (clang-diagnostic-return-type-c-linkage)

   /*
    * Legal values:
    * - from goes from 1 to len
    * - len goes from 0 to (strLen-from +1)
    *
    * illegal indices "count" towards the length;
    *    i.e. if we start at -1 with length 3 and the string has a length 10, the result has length 1
    * we then convert the value of from to from-1 to work with c++ structures
   */

   // length should not be negative
   int64_t legalizedLength = std::max(static_cast<int64_t>(0), len);
   // from should start at position 1.
   // from greater than size() will be truncated to size() by charIndexToByteIndex
   size_t legalizedFrom = std::max(from, static_cast<int64_t>(1));
   // legalizedTo should be at least legalizedFrom (outputting an empty string).
   // Note we work with the non-legalized from here, as semantically we can pass through empty indices before arriving at the actual string
   size_t legalizedTo = std::max(from + legalizedLength, static_cast<int64_t>(legalizedFrom));

   legalizedFrom--;
   legalizedTo--;

   size_t byteFrom = charIndexToByteIndex(str, legalizedFrom);
   size_t byteTo = charIndexToByteIndex(str, legalizedTo, byteFrom, legalizedFrom);

   return lingodb::runtime::VarLen32::fromString(str.str().substr(byteFrom, byteTo - byteFrom), StorageClass::REFCOUNTED);
}

// TODO add regexp flags
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::regexpReplace(
   VarLen32 text,
   VarLen32 pattern,
   VarLen32 replace) { // NOLINT (clang-diagnostic-return-type-c-linkage)

   // TODO handle more complex regexp featuers
   /*
    * SQL's regexpReplace works differently than C++'s regex_replace work differently e.g.
    * - group capturing uses \1 in sql and $1 in c++ (hence pre-parsing is necessary)
    * - the pattern .* returns different results
    */
   return VarLen32::fromString(std::regex_replace(text.str(), std::regex(pattern.str()), replace.str(), std::regex_constants::format_default), StorageClass::REFCOUNTED);
}

size_t lingodb::runtime::StringRuntime::findMatch(VarLen32 str, VarLen32 needle, size_t start, size_t end) {
   constexpr size_t invalidPos = 0x8000000000000000;
   if (start >= invalidPos) return invalidPos;
   if (start + needle.getLen() > end) return invalidPos;
   size_t found = std::string_view(str.data(), str.getLen()).find(std::string_view(needle.data(), needle.getLen()), start);

   if (found == std::string::npos || found + needle.getLen() > end) return invalidPos;
   return found + needle.getLen();
}
size_t lingodb::runtime::StringRuntime::findNext(VarLen32 str, VarLen32 needle, size_t start) {
   constexpr size_t invalidPos = 0x8000000000000000;
   if (start >= invalidPos) return invalidPos;
   size_t found = std::string_view(str.data(), str.getLen()).find(std::string_view(needle.data(), needle.getLen()), start);

   if (found == std::string::npos) return invalidPos;
   return found;
}
namespace {
void toUpper(char* str, size_t len) {
   for (auto i = 0ul; i < len; i++) {
      str[i] = std::toupper(str[i]);
   }
}
void toLower(char* str, size_t len) {
   for (auto i = 0ul; i < len; i++) {
      str[i] = std::tolower(str[i]);
   }
}

inline int64_t find(const std::string& str, const std::string& sub, int64_t start, int64_t end) {
   auto pos = str.find(sub, start);
   if (pos == std::string::npos || pos + sub.size() > (size_t) end) {
      return -1;
   } else {
      return pos;
   }
}

inline int64_t rfind(const std::string& str, const std::string& sub, int64_t start, int64_t end) {
   end -= sub.size();
   if (end < 0) end = 0;
   auto pos = str.rfind(sub, end);
   if (pos == std::string::npos || pos < (size_t) start) {
      return -1;
   } else {
      return pos;
   }
}
inline std::string replace(const std::string& str, const std::string& oldVal, const std::string& newVal) {
   auto len = oldVal.size();
   std::string output = str;
   size_t pos = output.find(oldVal);
   while (pos != std::string::npos) {
      output.replace(pos, len, newVal);
      pos = output.find(oldVal);
   }
   return output;
}
} // namespace
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::toUpper(lingodb::runtime::VarLen32 str) {
   if (str.isShort()) {
      ::toUpper(str.data(), str.getLen());
      return str;
   } else {
      char* copied = reinterpret_cast<char*>(VarLen32::allocateForStorageClass(str.getLen(), StorageClass::REFCOUNTED));

      memcpy(copied, str.data(), str.getLen());
      ::toUpper(copied, str.getLen());
      return lingodb::runtime::VarLen32(reinterpret_cast<uint8_t*>(copied), str.getLen(), StorageClass::REFCOUNTED);
   }
}
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::toLower(lingodb::runtime::VarLen32 str) {
   if (str.isShort()) {
      ::toLower(str.data(), str.getLen());
      return str;
   } else {
      char* copied = reinterpret_cast<char*>(VarLen32::allocateForStorageClass(str.getLen(), StorageClass::REFCOUNTED));
      memcpy(copied, str.data(), str.getLen());
      ::toLower(copied, str.getLen());
      return lingodb::runtime::VarLen32((uint8_t*) copied, str.getLen(), StorageClass::REFCOUNTED);
   }
}
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::concat(lingodb::runtime::VarLen32 a, lingodb::runtime::VarLen32 b) {
   auto totalLength = a.getLen() + b.getLen();
   if (totalLength <= lingodb::runtime::VarLen32::shortLen) {
      uint8_t data[lingodb::runtime::VarLen32::shortLen];
      memcpy(data, a.data(), a.getLen());
      memcpy(&data[a.getLen()], b.data(), b.getLen());
      return lingodb::runtime::VarLen32(data, totalLength, StorageClass::TRANSIENT);
   } else {
      auto* copied = VarLen32::allocateForStorageClass(totalLength, StorageClass::REFCOUNTED);
      memcpy(copied, a.data(), a.getLen());
      memcpy(&copied[a.getLen()], b.data(), b.getLen());
      return lingodb::runtime::VarLen32(copied, totalLength, StorageClass::REFCOUNTED);
   }
}

bool lingodb::runtime::StringRuntime::contains(VarLen32 str, VarLen32 substr) {
   if (str.getLen() < substr.getLen()) return false;
   return std::string_view(str.data(), str.getLen()).find(std::string_view(substr.data(), substr.getLen())) != std::string::npos;
}

int64_t lingodb::runtime::StringRuntime::toDate(lingodb::runtime::VarLen32 str) {
   int32_t res;
   arrow::internal::ParseValue<arrow::Date32Type>(str.data(), str.getLen(), &res);
   int64_t date64 = static_cast<int64_t>(res) * 24 * 60 * 60 * 1000000000ll;
   return date64;
}

int64_t lingodb::runtime::StringRuntime::toTimestamp(lingodb::runtime::VarLen32 str) {
   int64_t res;
   arrow::TimestampType t(arrow::TimeUnit::NANO);
   arrow::internal::ParseValue<arrow::TimestampType>(t, str.data(), str.getLen(), &res);
   return res;
}
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::fromDate(int64_t date) {
   static arrow_vendored::date::sys_days epoch = arrow_vendored::date::sys_days{arrow_vendored::date::jan / 1 / 1970};
   auto asString = arrow_vendored::date::format("%F", epoch + std::chrono::nanoseconds{date});
   return lingodb::runtime::VarLen32::fromString(asString, StorageClass::REFCOUNTED);
}
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::fromTimestamp(int64_t timestamp) {
   static arrow_vendored::date::sys_days epoch = arrow_vendored::date::sys_days{arrow_vendored::date::jan / 1 / 1970};
   auto asString = arrow_vendored::date::format("%F %T", epoch + std::chrono::nanoseconds{timestamp});
   return lingodb::runtime::VarLen32::fromString(asString, StorageClass::REFCOUNTED);
}

int32_t lingodb::runtime::StringRuntime::toChar(VarLen32 str) {
   assert(str.getLen() <= 4);
   return str.first4;
}

extern "C" lingodb::runtime::VarLen32 createVarLen32(uint8_t* ptr, uint32_t len) { //NOLINT(clang-diagnostic-return-type-c-linkage)
   return lingodb::runtime::VarLen32(ptr, len, lingodb::runtime::StorageClass::TRANSIENT);
}

int64_t lingodb::runtime::StringRuntime::pyFind(VarLen32 str, VarLen32 needle, int64_t start, int64_t end) {
   return find(str.str(), needle.str(), start, end);
}
int64_t lingodb::runtime::StringRuntime::pyRFind(VarLen32 str, VarLen32 needle, int64_t start, int64_t end) {
   return rfind(str.str(), needle.str(), start, end);
}
lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::replace(VarLen32 str, VarLen32 oldVal, VarLen32 newVal) {
   auto res = ::replace(str.str(), oldVal.str(), newVal.str());
   return lingodb::runtime::VarLen32::fromString(res, StorageClass::REFCOUNTED);
}

lingodb::runtime::List* lingodb::runtime::StringRuntime::split(VarLen32 str, VarLen32 needle, size_t maxSplits) {
   if (needle.getLen() == 0) {
      throw std::runtime_error("Cannot split by empty string");
   }
   auto* list = lingodb::runtime::List::create(sizeof(lingodb::runtime::VarLen32));
   size_t start = 0;
   size_t end = 0;
   size_t splits = 0;
   while (end < str.getLen()) {
      end = find(str.str(), needle.str(), start, str.getLen());
      if (end == static_cast<size_t>(-1)) {
         end = str.getLen();
      }
      if (splits >= maxSplits) {
         end = str.getLen();
      }
      auto* val = list->append();
      auto varlen = VarLen32::fromDataAndLen(str.data() + start, end - start, StorageClass::REFCOUNTED);
      *reinterpret_cast<VarLen32*>(val) = varlen;
      start = end + needle.getLen();
      splits++;
   }
   return list;
}

int64_t lingodb::runtime::StringRuntime::ord(VarLen32 str) {
   //convert utf8 char (1-4 bytes) to int32_t
   if (str.getLen() == 0) {
      throw std::runtime_error("Cannot get ord of empty string");
   }
   if (str.getLen() == 1) {
      return static_cast<int32_t>(str.data()[0]);
   }
   if (str.getLen() == 2) {
      return (static_cast<int32_t>(str.data()[0]) & 0x1F) << 6 | (static_cast<int32_t>(str.data()[1]) & 0x3F);
   }
   if (str.getLen() == 3) {
      return (static_cast<int32_t>(str.data()[0]) & 0x0F) << 12 | (static_cast<int32_t>(str.data()[1]) & 0x3F) << 6 | (static_cast<int32_t>(str.data()[2]) & 0x3F);
   }
   if (str.getLen() == 4) {
      return (static_cast<int32_t>(str.data()[0]) & 0x07) << 18 | (static_cast<int32_t>(str.data()[1]) & 0x3F) << 12 |
         (static_cast<int32_t>(str.data()[2]) & 0x3F) << 6 | (static_cast<int32_t>(str.data()[3]) & 0x3F);
   }
   throw std::runtime_error("Cannot get ord of string with length " + std::to_string(str.getLen()));
}

void lingodb::runtime::StringRuntime::cleanupUse(VarLen32 str) {
   lingodb::runtime::VarLen32::decRefCount(str);
}

void lingodb::runtime::StringRuntime::addUse(VarLen32 str) {
   lingodb::runtime::VarLen32::incRefCount(str);
}

lingodb::runtime::VarLen32 lingodb::runtime::StringRuntime::promoteToGlobal(lingodb::runtime::VarLen32 str) {
   return lingodb::runtime::VarLen32::promoteToGlobal(str);
}

bool lingodb::runtime::StringRuntime::likeProgramWithUnderscores(VarLen32 str, const int32_t* program, VarLen32 data){
   int32_t patternCount = *(program++);
   int32_t sums = program[2 + patternCount];
   int32_t size = str.getLen();
   if (size < sums) {
      return false;
   }
   const uint8_t* begin = str.getPtr();
   const uint8_t* end = begin + size;

   uint8_t* patterns = data.getPtr();
   int32_t prefixLen = *(program++);
   int32_t suffixLen = *(program++);
   const int32_t* twoWaySearchData = program + patternCount + 1;

   if (prefixLen != 0) {
      int32_t lengthUntilFirstUnderscore = *(twoWaySearchData++);
      if (memcmp(begin, patterns, lengthUntilFirstUnderscore) != 0) {
         return false;
      }
      begin += lengthUntilFirstUnderscore;
      int32_t index = lengthUntilFirstUnderscore;
      while (index < prefixLen) {
         uint8_t c = patterns[index];
         if (begin == end)
            return false;
         if (c == UNDERSCORE_REPLACEMENT) {
            begin += getLengthOfUTF8Sequence(*begin);
         } else if (c != *(begin++))
            return false;
         ++index;
      }

      patterns += prefixLen;
      sums -= prefixLen;
   }

   if (suffixLen != 0) {
      int32_t lengthAfterLastUnderscore = *(twoWaySearchData++);
      if ((end - begin) < sums)
         return false;
      if (memcmp(end - lengthAfterLastUnderscore, patterns + suffixLen - lengthAfterLastUnderscore, lengthAfterLastUnderscore) != 0) {
         return false;
      }
      int32_t index = suffixLen - lengthAfterLastUnderscore - 1;
      const uint8_t* iter = end - lengthAfterLastUnderscore - 1;
      while (index >= 0) {
         uint8_t c = patterns[index];
         if (iter < begin)
            return false;
         if (c == UNDERSCORE_REPLACEMENT) {
            iter = movePointerToCharacterStart(iter);
            --iter;
         }
         else if (c != *(iter--))
            return false;
         --index;
      }

      patterns += suffixLen;
      end = iter + 1;
      sums -= suffixLen;
   }

   for (int32_t index = 0; index < patternCount; ++index) {

      int32_t needleLen = *(program++);
      int32_t textLen = end - begin;

      if (textLen < sums)
         return false;

      int32_t numStartUnderscores = *(twoWaySearchData++);
      for (int32_t i = 0; i < numStartUnderscores; ++i) {
         if (begin == end) {
            return false;
         }
         begin += getLengthOfUTF8Sequence(*begin);
      }

      sums -= numStartUnderscores;
      patterns += numStartUnderscores;
      int32_t firstSubpatternLength = *(twoWaySearchData++);
      int32_t period = *(twoWaySearchData++);
      int32_t maxSuffix = *(twoWaySearchData++);
      if (firstSubpatternLength == 0) {
         continue;
      }
      textLen = end - begin;
      if (textLen < sums)
         return false;


      if (numStartUnderscores + firstSubpatternLength == needleLen) {
         int32_t searchLimit = textLen - sums + 1;
         const uint8_t* sep = twoWaySearch(begin, searchLimit, patterns, firstSubpatternLength, period, maxSuffix);
         if (!sep) {
            return false;
         }
         begin = sep + firstSubpatternLength;
         patterns += firstSubpatternLength;
         sums -= firstSubpatternLength;
      } else {
         begin = likeProgramWithUnderscoresStep(begin, end, patterns, patterns + needleLen - numStartUnderscores, firstSubpatternLength, sums, period, maxSuffix);
         if (!begin)
            return false;
         patterns += (needleLen - numStartUnderscores);
         sums -= (needleLen - numStartUnderscores);
      }
   }
   return true;
}

bool lingodb::runtime::StringRuntime::compareEqWithUnderscores(VarLen32 str, VarLen32 data){
   uint8_t* haystack = str.getPtr();
   uint8_t* haystackEnd = str.getPtr() + str.getLen();
   uint8_t* needle = data.getPtr();
   uint8_t* needleEnd = data.getPtr() + data.getLen();
   while (haystack < haystackEnd && needle < needleEnd) {
      if (*needle == UNDERSCORE_REPLACEMENT) {
         haystack += getLengthOfUTF8Sequence(*haystack);
      } else {
         if (*haystack != *needle) {
            return false;
         }
         ++haystack;
      }
      ++needle;
   }
   return haystack == haystackEnd && needle == needleEnd;
}

