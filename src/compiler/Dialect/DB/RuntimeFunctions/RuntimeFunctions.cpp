#include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h"

#include "lingodb/compiler/Dialect/DB/Passes.h"
#include "lingodb/compiler/runtime/DateRuntime.h"
#include "lingodb/compiler/runtime/DecimalRuntime.h"
#include "lingodb/compiler/runtime/DumpRuntime.h"
#include "lingodb/compiler/runtime/FloatRuntime.h"
#include "lingodb/compiler/runtime/IntegerRuntime.h"
#include "lingodb/compiler/runtime/StringRuntime.h"
#include "lingodb/compiler/runtime/Timing.h"
#include "lingodb/runtime/DateRuntime.h"
#include "lingodb/runtime/StringRuntime.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
lingodb::compiler::dialect::db::RuntimeFunction* lingodb::compiler::dialect::db::RuntimeFunctionRegistry::lookup(std::string name) {
   return registeredFunctions[name].get();
}
std::vector<std::basic_string<uint8_t>> lingodb::compiler::dialect::db::parseLikePattern(const std::string& pattern) {
   std::vector<std::basic_string<uint8_t>> subpatterns;
   std::basic_string<uint8_t> current;
   bool forceFlush = true;
   auto flushCurrent = [&]() {
      if (!current.empty() || forceFlush) {
         forceFlush = false;
         subpatterns.push_back(std::move(current));
         current.clear();
      }
   };

   size_t pos = 0;
   while (pos < pattern.size()) {
      char c = pattern[pos];
      if (c == '\\') {
         // this should never lead to a bug because of the verifier.
         current.push_back(pattern[++pos]);
      } else if (c == '%')
         flushCurrent();
      else if (c == '_') {
         current.push_back(UNDERSCORE_REPLACEMENT);
      } else
         current.push_back(c);
      ++pos;
   }
   forceFlush = true;
   flushCurrent();
   return subpatterns;
}

void lingodb::compiler::dialect::db::preprocessSubpattern(const uint8_t* subpattern, size_t size, int32_t& period, int32_t& maxSuffix) {
   if (size == 1) {
      maxSuffix = 0;
      period = 0;
      return;
   }

   if (size == 2) {
      if (subpattern[0] == subpattern[1]) {
         maxSuffix = -1;
         period = (3 << 1) | 1;
      } else {
         maxSuffix = 0;
         period = (1 << 1) | 0;
      }
      return;
   }

   // https://en.wikipedia.org/wiki/Two-way_string-matching_algorithm
   auto computeMaxSuffix = [&]<bool Reverse>() noexcept -> std::pair<int32_t, int32_t> {
      auto compare = [](uint8_t a, uint8_t b) -> int {
         int result = (a > b) - (b > a);
         if constexpr (Reverse) {
            return -result;
         } else {
            return result;
         }
      };

      int32_t currentPeriod = 1;
      int32_t maxSuffixIndex = -1;
      int32_t periodTestIndex = 1;
      int32_t maxSuffixTestIndex = 0;
      int32_t length = size;

      while (maxSuffixTestIndex + periodTestIndex < length) {
         int compareVal = compare(subpattern[maxSuffixTestIndex + periodTestIndex], subpattern[maxSuffixIndex + periodTestIndex]);
         if (compareVal < 0) {
            maxSuffixTestIndex += periodTestIndex;
            periodTestIndex = 1;
            currentPeriod = maxSuffixTestIndex - maxSuffixIndex;
         } else if (compareVal == 0) {
            if (periodTestIndex == currentPeriod) {
               maxSuffixTestIndex += currentPeriod;
               periodTestIndex = 1;
            } else {
               ++periodTestIndex;
            }
         } else {
            maxSuffixIndex = maxSuffixTestIndex;
            ++maxSuffixTestIndex;
            currentPeriod = 1;
            periodTestIndex = 1;
         }
      }
      return std::pair<int32_t, int32_t>{maxSuffixIndex, currentPeriod};
   };

   auto findCriticalFactorization = [&]() {
      auto [maxSuffixIndex1, period1] = computeMaxSuffix.template operator()<false>();
      auto [maxSuffixIndex2, period2] = computeMaxSuffix.template operator()<true>();
      return maxSuffixIndex1 > maxSuffixIndex2 ? std::pair<int, int>{maxSuffixIndex1, period1} : std::pair<int, int>{maxSuffixIndex2, period2};
   };

   auto result = findCriticalFactorization();
   maxSuffix = result.first;
   period = result.second;

   // function SMALL-PERIOD from http://monge.univ-mlv.fr/~mac/Articles-PDF/CP-1991-jacm.pdf
   // l represents the maxSuffix and p the period
   // x[1] x[2] ... x[l] is a suffix of x[l + 1] x[l + 2] ... x[l + p] if and only if x[1] x[2] ... x[l] == x[p + 1] x[p + 2] ... x[l + p]
   // we adjust this for 0-indexing rather than 0-indexing
   bool equal = !memcmp(subpattern, subpattern + period, maxSuffix + 1);

   if (!equal) {
      // Proposition 5.2 from http://monge.univ-mlv.fr/~mac/Articles-PDF/CP-1991-jacm.pdf
      // l represents the maxSuffix
      // adjusted because of 0-indexing rather than 1-indexing used in the paper
      // here, the period actually represents the value q, used for long-periods
      int32_t candidatePeriod1 = maxSuffix + 1;
      int32_t candidatePeriod2 = size - maxSuffix - 1;
      period = std::max(candidatePeriod1, candidatePeriod2) + 1;
   }
   period = (period << 1) | equal;
}

namespace {
using namespace lingodb::compiler::runtime;
using namespace lingodb::compiler::dialect;
mlir::Value dateAddImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   if (mlir::cast<db::IntervalType>(originalArgumentTypes[1]).getUnit() == db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::AddIOp>(loc, loweredArguments);
   } else {
      return DateRuntime::addMonths(rewriter, loc)(loweredArguments)[0];
   }
}
mlir::Value absImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   mlir::Value val = loweredArguments[0];
   mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, typeConverter->convertType(resType), rewriter.getIntegerAttr(typeConverter->convertType(resType), 0));
   mlir::Value negated = rewriter.create<mlir::arith::SubIOp>(loc, zero, val);
   mlir::Value ltZero = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, val, zero);
   return rewriter.create<mlir::arith::SelectOp>(loc, ltZero, negated, val);
}
mlir::Value sqrtImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;

   mlir::Value val = loweredArguments[0];
   if (mlir::isa<mlir::IntegerType>(val.getType())) {
      mlir::Value res = IntegerRuntime::sqrt(rewriter, loc)(val)[0]; //todo: for decimals
      if (res.getType() != val.getType()) {
         res = rewriter.create<mlir::arith::TruncIOp>(loc, val.getType(), res);
      }
      return res;

   } else {
      return FloatRuntime::sqrt(rewriter, loc)(val)[0];
   }
}
mlir::Value dateSubImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   if (mlir::cast<db::IntervalType>(originalArgumentTypes[1]).getUnit() == db::IntervalUnitAttr::daytime) {
      return rewriter.create<mlir::arith::SubIOp>(loc, loweredArguments);
   } else {
      return DateRuntime::subtractMonths(rewriter, loc)(loweredArguments)[0];
   }
}

mlir::Value constLikeWithUnderscoresImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   mlir::Value haystack = loweredArguments[0];
   mlir::Value patternValue = loweredArguments[1];
   // TODO: we had an if condition prior to this
   // TODO: however, can we get there with this if condition false?
   auto constStrOp = mlir::dyn_cast_or_null<util::CreateConstVarLen>(patternValue.getDefiningOp());
   auto subpatterns = lingodb::compiler::dialect::db::parseLikePattern(constStrOp.getStr().str());

   // pattern is of the form 'ABC_EF_HI_LM'
   if (subpatterns.size() == 1) {
      llvm::StringRef patternsString(reinterpret_cast<const char*>(subpatterns[0].data()), subpatterns[0].size());
      Value data = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(rewriter.getContext()), patternsString);
      return StringRuntime::compareEqWithUnderscores(rewriter, loc)({haystack, data})[0];
   }
   llvm::SmallVector<int32_t> patternProgram;
   llvm::SmallVector<uint8_t> bufferString;

   // the structure of the LIKE program is as follows:
   // first 4 bytes are the number of middle subpatterns
   // next 4 bytes are the size of the prefix (perhaps empty)
   // next 4 bytes are the size of the suffix (perhaps empty)
   patternProgram.push_back(subpatterns.size() - 2);

   auto& prefix = subpatterns.front();
   int32_t prefixSize = prefix.size();
   patternProgram.push_back(prefixSize);
   bufferString.append(prefix.begin(), prefix.end());

   auto& suffix = subpatterns.back();
   int32_t suffixSize = suffix.size();
   patternProgram.push_back(suffixSize);
   bufferString.append(suffix.begin(), suffix.end());

   int32_t sums = prefixSize + suffixSize;
   for (size_t i = 1; i < subpatterns.size() - 1; ++i) {
      auto& subpattern = subpatterns[i];
      int32_t size = subpattern.size();
      // then we enumerate the lengths of all middle subpatterns
      patternProgram.push_back(size);
      sums += size;
      bufferString.append(subpattern.begin(), subpattern.end());
   }
   // next 4 bytes are the total length of patterns (i.e., the string in bufferString)
   patternProgram.push_back(sums);

   if (prefixSize != 0) {
      int32_t i = 0;
      while (i < prefixSize && prefix[i] != UNDERSCORE_REPLACEMENT) {
         ++i;
      }
      patternProgram.push_back(i);
   }

   if (suffixSize != 0) {
      int32_t i = suffixSize - 1;
      while (i >= 0 && suffix[i] != UNDERSCORE_REPLACEMENT) {
         --i;
      }
      patternProgram.push_back(suffixSize - 1 - i);
   }

   for (size_t i = 1; i < subpatterns.size() - 1; ++i) {
      auto& subpattern = subpatterns[i];
      int32_t subpatternLength = subpattern.size();
      int32_t numStartUnderscores = 0;
      while (numStartUnderscores < subpatternLength && subpattern[numStartUnderscores] == UNDERSCORE_REPLACEMENT)
         ++numStartUnderscores;

      int32_t firstSubsubpatternLength = 0;
      int32_t maxSuffix = 0;
      int32_t period = 0;
      if (numStartUnderscores != subpatternLength) {
         int32_t ptr = numStartUnderscores;
         while (ptr < subpatternLength && subpattern[ptr] != UNDERSCORE_REPLACEMENT)
            ++ptr;
         firstSubsubpatternLength = ptr - numStartUnderscores;
         lingodb::compiler::dialect::db::preprocessSubpattern(subpattern.data() + numStartUnderscores, firstSubsubpatternLength, period, maxSuffix);
      }
      patternProgram.push_back(numStartUnderscores);
      patternProgram.push_back(firstSubsubpatternLength);
      patternProgram.push_back(period);
      patternProgram.push_back(maxSuffix);
   }

   auto shape = mlir::RankedTensorType::get({static_cast<int64_t>(patternProgram.size())}, rewriter.getI32Type());
   auto programAttr = mlir::DenseElementsAttr::get(shape, llvm::ArrayRef<int32_t>(patternProgram));
   Value program = rewriter.create<util::CreateConstArrayOp>(loc, util::RefType::get(rewriter.getI32Type()), programAttr);

   llvm::StringRef patternsString(reinterpret_cast<const char*>(bufferString.data()), bufferString.size());
   Value data = rewriter.create<util::CreateConstVarLen>(loc, util::VarLen32Type::get(rewriter.getContext()), patternsString);
   return StringRuntime::likeProgramWithUnderscores(rewriter, loc)({haystack, program, data})[0];
}
mlir::Value dumpValuesImpl(mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) {
   using namespace mlir;
   const auto i128Type = IntegerType::get(rewriter.getContext(), 128);
   const auto i64Type = IntegerType::get(rewriter.getContext(), 64);
   const auto i32Type = IntegerType::get(rewriter.getContext(), 32);
   const auto f64Type = Float64Type::get(rewriter.getContext());
   const auto nullableType = mlir::dyn_cast_or_null<db::NullableType>(originalArgumentTypes[0]);
   const auto baseType = getBaseType(originalArgumentTypes[0]);

   Value isNull;
   Value val;
   if (nullableType) {
      auto unPackOp = rewriter.create<util::UnPackOp>(loc, loweredArguments[0]);
      isNull = unPackOp.getVals()[0];
      val = unPackOp.getVals()[1];
   } else {
      isNull = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
      val = loweredArguments[0];
   }
   if (mlir::isa<mlir::IndexType>(baseType)) {
      DumpRuntime::dumpIndex(rewriter, loc)(loweredArguments[0]);
   } else if (isIntegerType(baseType, 1)) {
      DumpRuntime::dumpBool(rewriter, loc)({isNull, val});
   } else if (auto intWidth = getIntegerWidth(baseType, false)) {
      if (intWidth < 64) {
         val = rewriter.create<arith::ExtSIOp>(loc, i64Type, val);
      }
      DumpRuntime::dumpInt(rewriter, loc)({isNull, val});
   } else if (auto uIntWidth = getIntegerWidth(baseType, true)) {
      if (uIntWidth < 64) {
         val = rewriter.create<arith::ExtUIOp>(loc, i64Type, val);
      }
      DumpRuntime::dumpUInt(rewriter, loc)({isNull, val});
   } else if (auto decType = mlir::dyn_cast_or_null<db::DecimalType>(baseType)) {
      if (mlir::cast<mlir::IntegerType>(typeConverter->convertType(decType)).getWidth() < 128) {
         auto converted = rewriter.create<arith::ExtSIOp>(loc, rewriter.getIntegerType(128), val);
         val = converted;
      }
      Value low = rewriter.create<arith::TruncIOp>(loc, i64Type, val);
      Value shift = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i128Type, 64));
      Value scale = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(decType.getS()));
      Value high = rewriter.create<arith::ShRUIOp>(loc, i128Type, val, shift);
      high = rewriter.create<arith::TruncIOp>(loc, i64Type, high);
      DumpRuntime::dumpDecimal(rewriter, loc)({isNull, low, high, scale});
   } else if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(baseType)) {
      DumpRuntime::dumpDate(rewriter, loc)({isNull, val});
   } else if (auto timestampType = mlir::dyn_cast_or_null<db::TimestampType>(baseType)) {
      switch (timestampType.getUnit()) {
         case db::TimeUnitAttr::second: DumpRuntime::dumpTimestampSecond(rewriter, loc)({isNull, val}); break;
         case db::TimeUnitAttr::millisecond: DumpRuntime::dumpTimestampMilliSecond(rewriter, loc)({isNull, val}); break;
         case db::TimeUnitAttr::microsecond: DumpRuntime::dumpTimestampMicroSecond(rewriter, loc)({isNull, val}); break;
         case db::TimeUnitAttr::nanosecond: DumpRuntime::dumpTimestampNanoSecond(rewriter, loc)({isNull, val}); break;
      }
   } else if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(baseType)) {
      if (intervalType.getUnit() == db::IntervalUnitAttr::months) {
         DumpRuntime::dumpIntervalMonths(rewriter, loc)({isNull, val});
      } else {
         DumpRuntime::dumpIntervalDaytime(rewriter, loc)({isNull, val});
      }
   } else if (auto floatType = mlir::dyn_cast_or_null<mlir::FloatType>(baseType)) {
      if (floatType.getWidth() < 64) {
         val = rewriter.create<arith::ExtFOp>(loc, f64Type, val);
      }
      DumpRuntime::dumpFloat(rewriter, loc)({isNull, val});
   } else if (mlir::isa<db::StringType>(baseType)) {
      DumpRuntime::dumpString(rewriter, loc)({isNull, val});
   } else if (auto charType = mlir::dyn_cast_or_null<db::CharType>(baseType)) {
      Value numBytes = rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(charType.getLen()));
      if (charType.getLen() <= 1 && val.getType() != i32Type) {
         val = rewriter.create<arith::ExtSIOp>(loc, i32Type, val);
      }
      DumpRuntime::dumpChar(rewriter, loc)({isNull, val, numBytes});
   }
   return mlir::Value();
}
mlir::LogicalResult dateAddFoldFn(mlir::TypeRange types, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
   if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(types[0])) {
      if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(types[1])) {
         auto leftIntAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(operands[0]);
         auto rightIntAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(operands[1]);
         if (leftIntAttr && rightIntAttr) {
            if (intervalType.getUnit() == db::IntervalUnitAttr::daytime) {
               results.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(dateType.getContext(), 64), leftIntAttr.getValue() + rightIntAttr.getValue()));
               return mlir::success();
            } else {
               results.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(dateType.getContext(), 64), lingodb::runtime::DateRuntime::addMonths(leftIntAttr.getInt(), rightIntAttr.getInt())));
               return mlir::success();
            }
         }
      }
   }
   return mlir::failure();
}
mlir::LogicalResult dateSubtractFoldFn(mlir::TypeRange types, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult>& results) {
   if (auto dateType = mlir::dyn_cast_or_null<db::DateType>(types[0])) {
      if (auto intervalType = mlir::dyn_cast_or_null<db::IntervalType>(types[1])) {
         auto leftIntAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(operands[0]);
         auto rightIntAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(operands[1]);
         if (leftIntAttr && rightIntAttr) {
            if (intervalType.getUnit() == db::IntervalUnitAttr::daytime) {
               results.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(dateType.getContext(), 64), leftIntAttr.getValue() - rightIntAttr.getValue()));
               return mlir::success();
            } else {
               results.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(dateType.getContext(), 64), lingodb::runtime::DateRuntime::subtractMonths(leftIntAttr.getInt(), rightIntAttr.getInt())));
               return mlir::success();
            }
         }
      }
   }
   return mlir::failure();
}
mlir::LogicalResult constLikeFoldFn(mlir::TypeRange types, ::llvm::ArrayRef<mlir::Attribute> operands, ::llvm::SmallVectorImpl<mlir::OpFoldResult>& results) {
   auto haystackAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(operands[0]);
   auto needleAttr = mlir::dyn_cast_or_null<mlir::StringAttr>(operands[1]);
   if (!haystackAttr || !needleAttr) return mlir::failure();

   auto haystack = haystackAttr.getValue();
   auto needle = needleAttr.getValue();

   auto haystackVarLen = lingodb::runtime::VarLen32(reinterpret_cast<const uint8_t*>(haystack.data()), haystack.size(), lingodb::runtime::StorageClass::TRANSIENT);
   auto needleVarLen = lingodb::runtime::VarLen32(reinterpret_cast<const uint8_t*>(needle.data()), needle.size(), lingodb::runtime::StorageClass::TRANSIENT);
   bool result = lingodb::runtime::StringRuntime::like(haystackVarLen, needleVarLen);
   results.push_back(mlir::BoolAttr::get(types[0].getContext(), result));
   return mlir::success();
}
} // namespace
std::shared_ptr<db::RuntimeFunctionRegistry> db::RuntimeFunctionRegistry::getBuiltinRegistry(mlir::MLIRContext* context) {
   auto builtinRegistry = std::make_shared<RuntimeFunctionRegistry>(context);
   builtinRegistry->add("DumpValue").handlesNulls().matchesTypes({RuntimeFunction::anyType}, RuntimeFunction::noReturnType).implementedAs(dumpValuesImpl);
   auto resTypeIsI64 = [](mlir::Type t, mlir::TypeRange) { return t.isInteger(64); };
   auto resTypeIsF64 = [](mlir::Type t, mlir::TypeRange) { return t.isF64(); };
   auto resTypeIsBool = [](mlir::Type t, mlir::TypeRange) { return t.isInteger(1); };
   auto resTypeIsIndex = [](mlir::Type t, mlir::TypeRange) { return t.isIndex(); };
   builtinRegistry->add("Substring").implementedAs(StringRuntime::substr).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("StringFind").implementedAs(StringRuntime::findNext).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike, RuntimeFunction::intLike}, resTypeIsI64);
   builtinRegistry->add("StringLength").implementedAs(StringRuntime::len).matchesTypes({RuntimeFunction::stringLike}, resTypeIsI64);
   builtinRegistry->add("StringSplit").implementedAs(StringRuntime::split).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike, RuntimeFunction::intLike}, [](mlir::Type t, mlir::TypeRange) { return mlir::isa<db::ListType>(t) && mlir::isa<db::StringType>(mlir::cast<db::ListType>(t).getElementType()); });
   builtinRegistry->add("Ord").implementedAs(StringRuntime::ord).matchesTypes({RuntimeFunction::stringLike}, resTypeIsI64);

   builtinRegistry->add("ToUpper").implementedAs(StringRuntime::toUpper).matchesTypes({RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("ToLower").implementedAs(StringRuntime::toLower).matchesTypes({RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("Contains").implementedAs(StringRuntime::contains).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool);
   builtinRegistry->add("Concatenate").implementedAs(StringRuntime::concat).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("PyStringFind").implementedAs(StringRuntime::pyFind).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike, RuntimeFunction::intLike, RuntimeFunction::intLike}, resTypeIsI64);
   builtinRegistry->add("PyStringRFind").implementedAs(StringRuntime::pyRFind).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike, RuntimeFunction::intLike, RuntimeFunction::intLike}, resTypeIsI64);
   builtinRegistry->add("Replace").implementedAs(StringRuntime::replace).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike, RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());

   builtinRegistry->add("RegexpReplace").implementedAs(StringRuntime::regexpReplace).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike, RuntimeFunction::stringLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("Like").implementedAs(StringRuntime::like).matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool);
   builtinRegistry->add("ConstLikeUnderscores").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::stringLike}, resTypeIsBool).implementedAs(constLikeWithUnderscoresImpl).needsWrapping().folds(constLikeFoldFn);
   builtinRegistry->add("RoundDecimal").matchesTypes({RuntimeFunction::anyDecimal, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument()).needsWrapping().implementedAs([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) -> mlir::Value {
      mlir::Value s = rewriter.create<mlir::arith::ConstantIndexOp>(loc, mlir::cast<lingodb::compiler::dialect::db::DecimalType>(originalArgumentTypes[0]).getS());
      mlir::Value res = DecimalRuntime::round(rewriter, loc)(mlir::ValueRange({loweredArguments[0], loweredArguments[1], s}))[0];
      auto loweredResType = typeConverter->convertType(resType);
      if (!loweredResType.isInteger(128)) {
         res = rewriter.create<mlir::arith::TruncIOp>(loc, loweredResType, res);
      }
      return res;
   });
   builtinRegistry->add("RoundInt64").implementedAs(IntegerRuntime::round64).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("startTiming").implementedAs(Timing::start).matchesTypes({}, resTypeIsI64);
   builtinRegistry->add("startPerf").implementedAs(Timing::startPerf).matchesTypes({}, RuntimeFunction::noReturnType);
   builtinRegistry->add("stopPerf").implementedAs(Timing::stopPerf).matchesTypes({}, RuntimeFunction::noReturnType);
   builtinRegistry->add("stopTiming").implementedAs(Timing::stop).matchesTypes({RuntimeFunction::intLike}, RuntimeFunction::noReturnType);
   builtinRegistry->add("RoundInt32").implementedAs(IntegerRuntime::round32).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("RoundInt16").implementedAs(IntegerRuntime::round16).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("RoundInt8").implementedAs(IntegerRuntime::round8).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());
   builtinRegistry->add("RandomInRange").implementedAs(IntegerRuntime::randomInRange).matchesTypes({RuntimeFunction::intLike, RuntimeFunction::intLike}, RuntimeFunction::matchesArgument());

   /*
    * DateDiff and ExtractFromDate are special, as no implementation is provided through .implementedAs()
    * Instead during the pass OptimizeRuntimeFunctions they are optimized to call the function with the correct unit (e.g. Extract*Year*FromDate)
    */
   builtinRegistry->add("DateTrunc").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike}, RuntimeFunction::matchesArgument(1)).implementedAs(DateRuntime::dateTrunc);
   builtinRegistry->add("DateDiff").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).needsWrapping();
   builtinRegistry->add("DateDiffDay").matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::dateDiffDay);
   builtinRegistry->add("DateDiffHour").matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::dateDiffHour);
   builtinRegistry->add("DateDiffMinute").matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::dateDiffMinute);
   builtinRegistry->add("DateDiffSecond").matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::dateDiffSecond);
   builtinRegistry->add("ExtractFromDate").matchesTypes({RuntimeFunction::stringLike, RuntimeFunction::dateLike}, resTypeIsI64).needsWrapping();
   builtinRegistry->add("ExtractYearFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractYear);
   builtinRegistry->add("ExtractMonthFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractMonth);
   builtinRegistry->add("ExtractDayFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractDay);
   builtinRegistry->add("ExtractHourFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractHour);
   builtinRegistry->add("ExtractMinuteFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractMinute);
   builtinRegistry->add("ExtractSecondFromDate").matchesTypes({RuntimeFunction::dateLike}, resTypeIsI64).implementedAs(DateRuntime::extractSecond);
   builtinRegistry->add("DateAdd").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateAddImpl).folds(dateAddFoldFn);
   builtinRegistry->add("DateSubtract").handlesInvalid().matchesTypes({RuntimeFunction::dateLike, RuntimeFunction::dateInterval}, RuntimeFunction::matchesArgument()).implementedAs(dateSubImpl).folds(dateSubtractFoldFn);

   builtinRegistry->add("AbsInt").handlesInvalid().matchesTypes({RuntimeFunction::intLike}, RuntimeFunction::matchesArgument()).implementedAs(absImpl);
   builtinRegistry->add("AbsDecimal").handlesInvalid().matchesTypes({RuntimeFunction::anyDecimal}, RuntimeFunction::matchesArgument()).implementedAs(absImpl);
   builtinRegistry->add("Sqrt").needsWrapping().matchesTypes({RuntimeFunction::anyNumber}, RuntimeFunction::matchesArgument()).implementedAs(sqrtImpl);
   builtinRegistry->add("Sin").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::sin);
   builtinRegistry->add("Log").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::log);
   builtinRegistry->add("Exp").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::exp);
   builtinRegistry->add("Erf").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::erf);
   builtinRegistry->add("Cos").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::cos);
   builtinRegistry->add("ASin").matchesTypes({RuntimeFunction::float64}, resTypeIsF64).implementedAs(FloatRuntime::arcsin);
   builtinRegistry->add("Hash").matchesTypes({RuntimeFunction::anyType}, resTypeIsIndex).needsWrapping().implementedAs([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) -> mlir::Value {
      return rewriter.create<lingodb::compiler::dialect::db::Hash>(loc, loweredArguments[0]);
   });
   builtinRegistry->add("CombineHashes").matchesTypes({RuntimeFunction::onlyIndex, RuntimeFunction::onlyIndex}, resTypeIsIndex).needsWrapping().implementedAs([](mlir::OpBuilder& rewriter, mlir::ValueRange loweredArguments, mlir::TypeRange originalArgumentTypes, mlir::Type resType, const mlir::TypeConverter* typeConverter, mlir::Location loc) -> mlir::Value {
      return rewriter.create<util::HashCombine>(loc, rewriter.getIndexType(), loweredArguments[0], loweredArguments[1]);
   });
   return builtinRegistry;
}
