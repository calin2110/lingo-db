#include "lingodb/compiler/Conversion/UtilToLLVM/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <lingodb/compiler/Dialect/util/UtilOps.h>
#include "lingodb/compiler/helper.h"
#include "llvm/TargetParser/Host.h"
#include "lingodb/compiler/Dialect/DB/IR/RuntimeFunctions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace {
using namespace lingodb::compiler::dialect;
class SplitConstLike : public mlir::RewritePattern {
   public:
   SplitConstLike(mlir::MLIRContext* context) : RewritePattern(util::LikeOp::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto likeOp = mlir::cast<util::LikeOp>(op);
      auto loc = op->getLoc();
      int32_t sums = 0;
      if (mlir::StringAttr prefixAttr = likeOp.getPrefixAttr()) {
         sums += prefixAttr.size();
      }
      if (mlir::StringAttr suffixAttr = likeOp.getSuffixAttr()) {
         sums += suffixAttr.size();
      }

      for (mlir::Attribute subpattern: likeOp.getSubpatterns()) {
         auto strAttr = mlir::cast<mlir::StringAttr>(subpattern);
         sums += strAttr.size();
      }
      int32_t totalSums = sums;

      struct Step { mlir::Value cond, startIndex, endIndex; };
      mlir::Value sumsConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, sums);

      auto createOp = likeOp.getStr().getDefiningOp<util::CreateVarLen>();

      mlir::Value haystack = likeOp.getStr();
      mlir::Value haystackPtr = createOp ? createOp.getRef() : rewriter.create<util::VarLenGetRef>(loc, util::RefType::get(rewriter.getContext(), rewriter.getI8Type()), haystack);
      mlir::Value bytesVal = rewriter.create<util::VarLenGetInlinedString>(loc, rewriter.getIntegerType(128), haystack);

      llvm::SmallVector<std::function<Step(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>> inlinedSteps;
      llvm::SmallVector<std::function<Step(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>> ptrSteps;

      if (mlir::StringAttr prefixAttr = likeOp.getPrefixAttr()) {
         sums -= prefixAttr.size();
         ptrSteps.push_back([prefixAttr, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
            mlir::Value isMatch = opBuilder.create<util::StringStartsWith>(location, opBuilder.getI1Type(), haystackPtr, prefixAttr, startIndex, endIndex);
            mlir::Value prefixSizeConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, prefixAttr.size());
            mlir::Value newStartIndex = opBuilder.create<mlir::arith::AddIOp>(location, startIndex, prefixSizeConst);
            return {isMatch, newStartIndex, endIndex};
         });

         inlinedSteps.push_back([prefixAttr, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
            mlir::Value isMatch = opBuilder.create<util::BytesStartsWith>(location, opBuilder.getI1Type(), bytesVal, prefixAttr, startIndex, endIndex);
            mlir::Value prefixSizeConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, prefixAttr.size());
            mlir::Value newStartIndex = opBuilder.create<mlir::arith::AddIOp>(location, startIndex, prefixSizeConst);
            return {isMatch, newStartIndex, endIndex};
         });
      }

      if (mlir::StringAttr suffixAttr = likeOp.getSuffixAttr()) {
         sums -= suffixAttr.size();
         ptrSteps.push_back([suffixAttr, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
            mlir::Value isMatch = opBuilder.create<util::StringEndsWith>(location, opBuilder.getI1Type(), haystackPtr, suffixAttr, startIndex, endIndex);
            mlir::Value suffixSizeConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, suffixAttr.size());
            mlir::Value newEndIndex = opBuilder.create<mlir::arith::SubIOp>(location, endIndex, suffixSizeConst);
            return {isMatch, startIndex, newEndIndex};
         });
         inlinedSteps.push_back([suffixAttr, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
            mlir::Value isMatch = opBuilder.create<util::BytesEndsWith>(location, opBuilder.getI1Type(), bytesVal, suffixAttr, startIndex, endIndex);
            mlir::Value suffixSizeConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, suffixAttr.size());
            mlir::Value newEndIndex = opBuilder.create<mlir::arith::SubIOp>(location, endIndex, suffixSizeConst);
            return {isMatch, startIndex, newEndIndex};
         });
      }

      for (mlir::Attribute attr: likeOp.getSubpatterns()) {
         auto subpatternAttr = mlir::cast<mlir::StringAttr>(attr);
         sums -= subpatternAttr.size();

         ptrSteps.push_back([sums, subpatternAttr, &haystackPtr](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
            mlir::Value sumsValue = opBuilder.create<mlir::arith::ConstantIndexOp>(location, sums);
            mlir::Value earlyStopEndIndex = opBuilder.create<mlir::arith::SubIOp>(location, endIndex, sumsValue);
            auto containsRet = opBuilder.create<util::StringContains>(location, opBuilder.getI1Type(), opBuilder.getIndexType(), haystackPtr, subpatternAttr, startIndex, earlyStopEndIndex);
            mlir::Value subpatternSizeConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, subpatternAttr.size());
            mlir::Value newStartIndex = opBuilder.create<mlir::arith::AddIOp>(location, containsRet.getMatchStart(), subpatternSizeConst);
            return {containsRet.getContains(), newStartIndex, endIndex};
         });

         inlinedSteps.push_back([sums, subpatternAttr, &bytesVal](mlir::OpBuilder& opBuilder, mlir::Location location, mlir::Value startIndex, mlir::Value endIndex) -> Step {
            mlir::Value sumsValue = opBuilder.create<mlir::arith::ConstantIndexOp>(location, sums);
            mlir::Value earlyStopEndIndex = opBuilder.create<mlir::arith::SubIOp>(location, endIndex, sumsValue);
            auto containsRet = opBuilder.create<util::BytesContains>(location, opBuilder.getI1Type(), opBuilder.getIndexType(), bytesVal, subpatternAttr, startIndex, earlyStopEndIndex);
            mlir::Value subpatternSizeConst = opBuilder.create<mlir::arith::ConstantIndexOp>(location, subpatternAttr.size());
            mlir::Value newStartIndex = opBuilder.create<mlir::arith::AddIOp>(location, containsRet.getMatchStart(), subpatternSizeConst);
            return {containsRet.getContains(), newStartIndex, endIndex};
         });
      }

      std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, size_t, mlir::Value, mlir::Value, llvm::SmallVector<std::function<Step(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>>&)> emit = [&](mlir::OpBuilder& opBuilder, mlir::Location location, size_t i, mlir::Value startIndex, mlir::Value endIndex, llvm::SmallVector<std::function<Step(mlir::OpBuilder&, mlir::Location, mlir::Value, mlir::Value)>>& steps) -> mlir::Value {
         if (i == steps.size())
            return opBuilder.create<mlir::arith::ConstantIntOp>(location, true, opBuilder.getI1Type());

         Step step = steps[i](opBuilder, location, startIndex, endIndex);
         return opBuilder.create<mlir::scf::IfOp>(location, step.cond, [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
            opBuilder2.create<mlir::scf::YieldOp>(location2, emit(opBuilder2, location2, i + 1, step.startIndex, step.endIndex, steps));
         }, [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
            opBuilder2.create<mlir::scf::YieldOp>(location2, opBuilder2.create<mlir::arith::ConstantIntOp>(location2, false, opBuilder2.getI1Type()).getResult());
         }).getResult(0);
      };

      mlir::Value startIndex = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value endIndex = rewriter.create<util::VarLenGetLen>(loc, rewriter.getIndexType(), haystack);
      mlir::Value canMatch = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, endIndex, sumsConst);
      auto ifBlock = rewriter.create<mlir::scf::IfOp>(loc,canMatch,
         [&](mlir::OpBuilder& opBuilder, mlir::Location location) {
            mlir::Value partialResult;
            if (createOp || totalSums > 12) {
               partialResult = emit(opBuilder, location, 0, startIndex, endIndex, ptrSteps);
            } else {
               mlir::Value inlineThreshold = opBuilder.create<mlir::arith::ConstantIndexOp>(location, 12);
               mlir::Value isNotInlined = opBuilder.create<mlir::arith::CmpIOp>(location, mlir::arith::CmpIPredicate::ugt, endIndex, inlineThreshold);

               auto ifOp = opBuilder.create<mlir::scf::IfOp>(location, isNotInlined,
                  [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                     opBuilder2.create<mlir::scf::YieldOp>(location2, emit(opBuilder2, location2, 0, startIndex, endIndex, ptrSteps));
                  },
                  [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                     opBuilder2.create<mlir::scf::YieldOp>(location2, emit(opBuilder2, location2, 0, startIndex, endIndex, inlinedSteps));
                  });
               partialResult = ifOp.getResult(0);
            }
            opBuilder.create<mlir::scf::YieldOp>(location, partialResult);
         },
         [&](mlir::OpBuilder& opBuilder, mlir::Location location) {
            opBuilder.create<mlir::scf::YieldOp>(location, opBuilder.create<mlir::arith::ConstantIntOp>(location, false, 1).getResult());
         });

      mlir::Value result = ifBlock.getResult(0);
      rewriter.replaceOp(op, result);
      return mlir::success(true);
   }
};


class SpecializeStringContains : public mlir::RewritePattern {
   public:
   SpecializeStringContains(mlir::MLIRContext* context) : RewritePattern(util::StringContains::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto containsOp = mlir::cast<util::StringContains>(op);
      llvm::StringRef needle = containsOp.getSubpattern();
      auto loc = containsOp.getLoc();
      auto i1Type = rewriter.getI1Type();
      auto indexType = rewriter.getIndexType();
      auto i8Type = rewriter.getI8Type();
      auto i8RefType = util::RefType::get(rewriter.getContext(), rewriter.getI8Type());

      mlir::Value haystackPtr = containsOp.getStr();
      mlir::Value startIdx = containsOp.getStartIndex();
      mlir::Value endIdx = containsOp.getEndIndex();
      mlir::Value stringLen = rewriter.create<mlir::arith::SubIOp>(loc, indexType, endIdx, startIdx);
      mlir::Value found;
      mlir::Value pos;
      if (needle.size() == 1) {
         mlir::Value memchrStart = rewriter.create<util::ArrayElementPtrOp>(loc, i8RefType, haystackPtr, startIdx);
         auto result = rewriter.create<util::RefMemchr>(loc, i1Type, indexType, memchrStart, stringLen, rewriter.getI8IntegerAttr(needle[0]));
         found = result.getResult(0);
         pos = rewriter.create<mlir::arith::AddIOp>(loc, result.getResult(1), startIdx);
      } else {
         mlir::Value statusRunning = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(0));
         mlir::Value statusFound   = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(1));
         mlir::Value statusFailed  = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(2));
         mlir::Value oneConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
         mlir::Value needleLenMinusOne  = rewriter.create<mlir::arith::ConstantIndexOp>(loc, needle.size() - 1);
         mlir::Value maxWhileIndex = rewriter.create<mlir::arith::SubIOp>(loc, indexType, endIdx, needleLenMinusOne);
         mlir::Value needleLenConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, needle.size());


         auto needleShape = mlir::RankedTensorType::get({static_cast<int64_t>(needle.size())}, rewriter.getI8Type());
         auto needleAttr = mlir::DenseElementsAttr::get(needleShape, llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t*>(needle.data()), needle.size()));
         mlir::Value needleGlobal = rewriter.create<util::CreateConstArrayOp>(loc, util::RefType::get(rewriter.getI8Type()), needleAttr);

         auto twoWaySearch = [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
            auto whileOp = opBuilder2.create<mlir::scf::WhileOp>(location2, mlir::TypeRange{indexType, i8Type},
               mlir::ValueRange{startIdx, statusRunning},
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3, mlir::ValueRange args) {
                  mlir::Value status = args[1];
                  mlir::Value running = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::eq, status, statusRunning);
                  opBuilder3.create<mlir::scf::ConditionOp>(location3, running, args);
               },
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3, mlir::ValueRange args) {
                  mlir::Value i = args[0];
                  mlir::Value memchrStart = opBuilder3.create<util::ArrayElementPtrOp>(location3, i8RefType, haystackPtr, i);
                  mlir::Value memchrLen = opBuilder3.create<mlir::arith::SubIOp>(location3, indexType, maxWhileIndex, i);
                  auto memchrRes = opBuilder3.create<util::RefMemchr>(location3, i1Type, indexType, memchrStart, memchrLen, opBuilder3.getI8IntegerAttr(needle[0]));
                  mlir::Value foundPartialMatch = memchrRes.getResult(0);
                  mlir::Value indexMatch = memchrRes.getResult(1);
                  indexMatch = opBuilder3.create<mlir::arith::AddIOp>(location3, indexType, i, indexMatch);

                  auto outer = opBuilder3.create<mlir::scf::IfOp>(location3, foundPartialMatch,
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4) {
                        mlir::Value nextNeedleCharConst = opBuilder4.create<mlir::arith::ConstantOp>(location4, opBuilder4.getI8IntegerAttr(needle[1]));
                        mlir::Value nextPos = opBuilder4.create<mlir::arith::AddIOp>(location4, indexType, indexMatch, oneConst);
                        mlir::Value ptrPos = opBuilder4.create<util::ArrayElementPtrOp>(location4, i8RefType, haystackPtr, nextPos);
                        mlir::Value nextByte = opBuilder4.create<util::LoadOp>(location4, i8Type, ptrPos);
                        mlir::Value isMatch = opBuilder4.create<mlir::arith::CmpIOp>(location4, mlir::arith::CmpIPredicate::eq, nextByte, nextNeedleCharConst);

                        auto inner = opBuilder4.create<mlir::scf::IfOp>(location4, isMatch,
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5) {
                              opBuilder5.create<mlir::scf::YieldOp>(location5, mlir::ValueRange{indexMatch, statusFound});
                           },
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5) {
                              mlir::Value atEnd;
                              if (needle[0] == needle[1]) {
                                 nextPos = opBuilder5.create<mlir::arith::AddIOp>(location5, indexType, nextPos, oneConst);
                                 atEnd = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::uge, nextPos, maxWhileIndex);
                              } else {
                                 atEnd = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::eq, nextPos, maxWhileIndex);
                              }
                              mlir::Value nextStatus = opBuilder5.create<mlir::arith::SelectOp>(location5, atEnd, statusFailed, statusRunning);
                                 opBuilder5.create<mlir::scf::YieldOp>(location5, mlir::ValueRange{nextPos, nextStatus});
                        });

                        opBuilder4.create<mlir::scf::YieldOp>(location4, inner.getResults());
                     },
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4) {
                        opBuilder4.create<mlir::scf::YieldOp>(location4, mlir::ValueRange{i, statusFailed});
                  });
                  opBuilder3.create<mlir::scf::YieldOp>(location3, outer.getResults());
            });

            int32_t period;
            int32_t maxSuffix;
            db::preprocessSubpattern(reinterpret_cast<const uint8_t*>(needle.data()), needle.size(), period, maxSuffix);
            bool isEqual = period & 1;
            period >>= 1;
            mlir::Value mayContinue = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::eq, whileOp.getResult(1), statusFound);
            auto ifBlock = opBuilder2.create<mlir::scf::IfOp>(location2, mayContinue,
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  mlir::Value periodConst= opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, period);
                  mlir::Value maxSufConst= opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, maxSuffix + 1);
                  mlir::Value resetPtrConst= opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, needle.size() - period);
                  mlir::Value minusOneIndex = opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, -1);
                  mlir::Value zeroConst = opBuilder3.create<mlir::arith::ConstantIndexOp>(location3, 0);

                  auto newWhileOp = opBuilder3.create<mlir::scf::WhileOp>(location3, mlir::TypeRange{indexType, indexType, i8Type}, mlir::ValueRange{whileOp.getResult(0), zeroConst, statusRunning},
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4, mlir::ValueRange args) {
                        mlir::Value running = opBuilder4.create<mlir::arith::CmpIOp>(location4, mlir::arith::CmpIPredicate::eq, args[2], statusRunning);
                        mlir::Value inRange = opBuilder4.create<mlir::arith::CmpIOp>(location4, mlir::arith::CmpIPredicate::ult, args[0], maxWhileIndex);
                        opBuilder4.create<mlir::scf::ConditionOp>(location4, opBuilder4.create<mlir::arith::AndIOp>(location4, running, inRange), args);
                     },
                     [&](mlir::OpBuilder& opBuilder4, mlir::Location location4, mlir::ValueRange args) {
                        mlir::Value offset = args[0];
                        mlir::Value lastPtr = args[1];
                        mlir::Value startPos;
                        if (!isEqual) {
                           startPos = maxSufConst;
                        } else {
                           startPos = opBuilder4.create<mlir::arith::MaxUIOp>(location4, lastPtr, maxSufConst);
                        }
                        auto fwd = opBuilder4.create<mlir::scf::WhileOp>(location4, mlir::TypeRange{indexType}, mlir::ValueRange{startPos},
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5, mlir::ValueRange args2) {
                              mlir::Value currentPos = args2[0];
                              mlir::Value inBounds = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::ult, currentPos, needleLenConst);
                              // short-circuit: only load when pos is in bounds
                              auto guard = opBuilder5.create<mlir::scf::IfOp>(location5, inBounds,
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6) {
                                    mlir::Value needleByte = opBuilder6.create<util::GetConstArrayAtOp>(location6, i8Type, needleGlobal, currentPos);
                                    mlir::Value haystackPos = opBuilder6.create<mlir::arith::AddIOp>(location6, indexType, currentPos, offset);
                                    mlir::Value currentHaystackPtr = opBuilder6.create<util::ArrayElementPtrOp>(location6, i8RefType, haystackPtr, haystackPos);
                                    mlir::Value haystackByte = opBuilder6.create<util::LoadOp>(location6, i8Type, currentHaystackPtr);
                                    mlir::Value areEqual = opBuilder6.create<mlir::arith::CmpIOp>(location6, mlir::arith::CmpIPredicate::eq, needleByte, haystackByte);
                                    opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{areEqual});
                                 },
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6) {
                                    mlir::Value falseConst = opBuilder6.create<mlir::arith::ConstantOp>(location6, opBuilder6.getBoolAttr(false));
                                    opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{falseConst});
                              });

                              opBuilder5.create<mlir::scf::ConditionOp>(location5, guard.getResult(0), args2);
                           },

                           // after: pos++
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5, mlir::ValueRange args2) {
                              mlir::Value next = opBuilder5.create<mlir::arith::AddIOp>(location5, indexType, args2[0], oneConst);
                              opBuilder5.create<mlir::scf::YieldOp>(location5, mlir::ValueRange{next});
                        });
                        mlir::Value forwardComplete = opBuilder4.create<mlir::arith::CmpIOp>(location4, mlir::arith::CmpIPredicate::eq, fwd.getResult(0), needleLenConst);
                        auto branchingPaths = opBuilder4.create<mlir::scf::IfOp>(location4, forwardComplete,
                           [&](mlir::OpBuilder& opBuilder5, mlir::Location location5) {
                              mlir::Value maxSufMinusOne = opBuilder5.create<mlir::arith::ConstantIndexOp>(location5, maxSuffix);
                              auto bwd = opBuilder5.create<mlir::scf::WhileOp>(location5, mlir::TypeRange{indexType}, mlir::ValueRange{maxSufMinusOne},
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6, mlir::ValueRange args2) {
                                    mlir::Value currentPos = args2[0];
                                    // TODO: is this correct?
                                    mlir::Value inBounds;
                                    if (isEqual) {
                                       inBounds = opBuilder6.create<mlir::arith::CmpIOp>(location6, mlir::arith::CmpIPredicate::sge, currentPos, lastPtr);
                                    } else {
                                       inBounds = opBuilder6.create<mlir::arith::CmpIOp>(location6, mlir::arith::CmpIPredicate::ne, currentPos, minusOneIndex);
                                    }
                                    auto guard = opBuilder6.create<mlir::scf::IfOp>(location6, inBounds,
                                       [&](mlir::OpBuilder& opBuilder7, mlir::Location location7) {
                                          mlir::Value needleByte = opBuilder7.create<util::GetConstArrayAtOp>(location7, i8Type, needleGlobal, currentPos);
                                          mlir::Value haystackPos = opBuilder7.create<mlir::arith::AddIOp>(location7, indexType, currentPos, offset);
                                          mlir::Value currentHaystackPtr = opBuilder7.create<util::ArrayElementPtrOp>(location7, i8RefType, haystackPtr, haystackPos);
                                          mlir::Value haystackByte = opBuilder7.create<util::LoadOp>(location7, i8Type, currentHaystackPtr);
                                          mlir::Value areEqual = opBuilder7.create<mlir::arith::CmpIOp>(location7, mlir::arith::CmpIPredicate::eq, needleByte, haystackByte);
                                          opBuilder7.create<mlir::scf::YieldOp>(location7, mlir::ValueRange{areEqual});
                                       },
                                       [&](mlir::OpBuilder& opBuilder7, mlir::Location location7) {
                                          mlir::Value falseConst = opBuilder7.create<mlir::arith::ConstantOp>(location7, opBuilder7.getBoolAttr(false));
                                          opBuilder7.create<mlir::scf::YieldOp>(location7, mlir::ValueRange{falseConst});
                                    });
                                    opBuilder6.create<mlir::scf::ConditionOp>(location6, guard.getResult(0), args2);
                              },
                              [&](mlir::OpBuilder& opBuilder6, mlir::Location location6, mlir::ValueRange args2) {
                                 mlir::Value prev = opBuilder6.create<mlir::arith::SubIOp>(location6, indexType, args2[0], oneConst);
                                 opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{prev});
                              });

                              mlir::Value stillMismatch;
                              if (isEqual) {
                                 stillMismatch = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::sge, bwd.getResult(0), lastPtr);
                              } else {
                                 stillMismatch = opBuilder5.create<mlir::arith::CmpIOp>(location5, mlir::arith::CmpIPredicate::ne, bwd.getResult(0), minusOneIndex);
                              }

                              auto inner = opBuilder5.create<mlir::scf::IfOp>(location5, stillMismatch,
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6) {
                                    mlir::Value next = opBuilder6.create<mlir::arith::AddIOp>(location6, indexType, offset, periodConst);
                                    opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{next, resetPtrConst, statusRunning});
                                 },
                                 [&](mlir::OpBuilder& opBuilder6, mlir::Location location6) {
                                    opBuilder6.create<mlir::scf::YieldOp>(location6, mlir::ValueRange{offset, lastPtr, statusFound});
                              });

                              opBuilder5.create<mlir::scf::YieldOp>(location5, inner.getResults());
                        },
                        [&](mlir::OpBuilder& opBuilder5, mlir::Location location5) {
                           mlir::Value addOne = opBuilder5.create<mlir::arith::AddIOp>(location5, indexType, fwd.getResult(0), oneConst);
                           mlir::Value delta = opBuilder5.create<mlir::arith::SubIOp>(location5, indexType, addOne, maxSufConst);
                           mlir::Value next = opBuilder5.create<mlir::arith::AddIOp>(location5, indexType, offset, delta);
                           opBuilder5.create<mlir::scf::YieldOp>(location5, mlir::ValueRange{next, zeroConst, statusRunning});
                        });
                        opBuilder4.create<mlir::scf::YieldOp>(location4, branchingPaths.getResults());
                  });
                  mlir::Value innerFound = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::eq, newWhileOp.getResult(2), statusFound);
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{innerFound, newWhileOp.getResult(0)});
               },
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{opBuilder3.create<mlir::arith::ConstantOp>(location3, opBuilder3.getBoolAttr(false)), whileOp.getResult(0)});
            });
            return ifBlock;
         };

         mlir::Value const16 = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 16);
         auto simdSearch = [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
            mlir::Value needleVec;
            auto v16i8 = mlir::VectorType::get({16}, i8Type);

            {
               uint8_t buf[16] = {0};
               memcpy(buf, needle.data(), needle.size());
               auto needleVecAttr = mlir::DenseElementsAttr::get(v16i8, llvm::ArrayRef<uint8_t>(buf, 16));
               needleVec =  opBuilder2.create<mlir::arith::ConstantOp>(location2, needleVecAttr);
            }

            mlir::Value lastFullMatchIndex = opBuilder2.create<mlir::arith::ConstantOp>(location2, rewriter.getI32IntegerAttr(16 - needle.size()));
            // _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED
            auto cmpistriFlags = opBuilder2.getI8IntegerAttr(0x0C);
            mlir::Value endIdxMinus16 = opBuilder2.create<mlir::arith::SubIOp>(location2, endIdx, const16);
            auto whileOp = opBuilder2.create<mlir::scf::WhileOp>(location2, mlir::TypeRange{indexType, i8Type}, mlir::ValueRange{startIdx, statusRunning},
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3, mlir::ValueRange args) {
                  mlir::Value offset = args[0];
                  mlir::Value status = args[1];
                  mlir::Value running = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::eq, status, statusRunning);
                  mlir::Value inBounds = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::ule, offset, endIdxMinus16);
                  mlir::Value cond = opBuilder3.create<mlir::arith::AndIOp>(location3, running, inBounds);
                  opBuilder3.create<mlir::scf::ConditionOp>(location3, cond, args);
               },
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3, mlir::ValueRange args) {
                  mlir::Value offset = args[0];
                  mlir::Value currentHaystackPtr = opBuilder3.create<util::ArrayElementPtrOp>(location3, i8RefType, haystackPtr, offset);
                  mlir::Value haystackVec = opBuilder3.create<util::LoadVectorOp>(location3, v16i8, currentHaystackPtr, opBuilder3.getI64IntegerAttr(1));
                  mlir::Value cmpistriRes = opBuilder3.create<util::CmpistriOp>(location3, opBuilder3.getI32Type(), needleVec, haystackVec, cmpistriFlags);
                  mlir::Value mayReturn = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::ule, cmpistriRes, lastFullMatchIndex);
                  mlir::Value asIndex = opBuilder3.create<mlir::arith::IndexCastOp>(location3, indexType, cmpistriRes);
                  mlir::Value matchStart = opBuilder3.create<mlir::arith::AddIOp>(location3, indexType, offset, asIndex);
                  mlir::Value nextStatus = opBuilder3.create<mlir::arith::SelectOp>(location3, mayReturn, statusFound, statusRunning);
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{matchStart, nextStatus});
            });

            auto isFinished = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::eq, whileOp.getResult(1), statusFound);
            auto ifOp = opBuilder2.create<mlir::scf::IfOp>(location2,isFinished,
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  mlir::Value trueConst = opBuilder3.create<mlir::arith::ConstantOp>(location3, opBuilder3.getBoolAttr(true));
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{trueConst, whileOp.getResult(0)});
               },
               [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                  mlir::Value offset = endIdxMinus16;
                  mlir::Value currentHaystackPtr = opBuilder3.create<util::ArrayElementPtrOp>(location3, i8RefType, haystackPtr, offset);
                  mlir::Value haystackVec = opBuilder3.create<util::LoadVectorOp>(location3, v16i8, currentHaystackPtr, opBuilder3.getI64IntegerAttr(1));
                  mlir::Value cmpistriRes = opBuilder3.create<util::CmpistriOp>(location3,opBuilder3.getI32Type(),  needleVec, haystackVec, cmpistriFlags);
                  mlir::Value mayReturn = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::ule, cmpistriRes, lastFullMatchIndex);
                  mlir::Value asIndex = opBuilder3.create<mlir::arith::IndexCastOp>(location3, indexType, cmpistriRes);
                  mlir::Value matchStart = opBuilder3.create<mlir::arith::AddIOp>(location3, indexType, offset, asIndex);
                  opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{mayReturn, matchStart});
            });
            return ifOp;
         };

         auto canUseSSE42 = []() {
            auto features = llvm::sys::getHostCPUFeatures();
            auto it = features.find("sse4.2");
            return it != features.end() && it->second;
         };


         if (!canUseSSE42() || needle.size() > 12) {
            auto result = twoWaySearch(rewriter, loc);
            found = result.getResult(0);
            pos = result.getResult(1);
         } else {
            mlir::Value isLongEnough = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, stringLen, const16);
            auto ifOp = rewriter.create<mlir::scf::IfOp>(loc,isLongEnough,
               [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                  auto result = simdSearch(opBuilder2, location2);
                  opBuilder2.create<mlir::scf::YieldOp>(location2, mlir::ValueRange{result.getResult(0), result.getResult(1)});
               },
               [&](mlir::OpBuilder& opBuilder2, mlir::Location location2) {
                  auto result = twoWaySearch(opBuilder2, location2);
                  opBuilder2.create<mlir::scf::YieldOp>(location2, mlir::ValueRange{result.getResult(0), result.getResult(1)});
               });
            found = ifOp.getResult(0);
         }
      }
      rewriter.replaceOp(containsOp, mlir::ValueRange{found, pos});
      return mlir::success();
   }
};

class SpecializeBytesContains : public mlir::RewritePattern {
   public:
   SpecializeBytesContains(mlir::MLIRContext* context) : RewritePattern(util::BytesContains::getOperationName(), 1, context) {}

   mlir::LogicalResult matchAndRewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const override {
      auto containsOp = mlir::cast<util::BytesContains>(op);
      llvm::StringRef needle = containsOp.getSubpattern();
      auto loc = containsOp.getLoc();
      auto i1Type = rewriter.getI1Type();
      auto indexType = rewriter.getIndexType();
      auto i8Type = rewriter.getI8Type();
      auto i8RefType = util::RefType::get(rewriter.getContext(), rewriter.getI8Type());
      auto i128Type = rewriter.getIntegerType(128);

      mlir::Value haystackInlined = containsOp.getStr();
      mlir::Value startIdx = containsOp.getStartIndex();
      mlir::Value endIdx = containsOp.getEndIndex();
      mlir::Value stringLen = rewriter.create<mlir::arith::SubIOp>(loc, indexType, endIdx, startIdx);
      mlir::Value found;
      mlir::Value pos;
      if (needle.size() == 1) {
         mlir::Value eight = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 8);
         mlir::Value bitOffIdx = rewriter.create<mlir::arith::MulIOp>(loc, startIdx, eight);
         mlir::Value bitOff = rewriter.create<mlir::arith::IndexCastOp>(loc, i128Type, bitOffIdx);
         mlir::Value shiftedString = rewriter.create<mlir::arith::ShRUIOp>(loc, haystackInlined, bitOff);
         auto result = rewriter.create<util::InlineMemchr>(loc, i1Type, indexType, shiftedString, stringLen, rewriter.getI8IntegerAttr(needle[0]));
         found = result.getResult(0);
         pos = rewriter.create<mlir::arith::AddIOp>(loc, indexType, startIdx, result.getResult(1));
      } else {
          uint64_t needleBits[2] = {0};
          uint8_t* needleBitsU8 = reinterpret_cast<uint8_t*>(needleBits);
          uint64_t maskBits[2] = {0};
          uint8_t* maskBitsU8 = reinterpret_cast<uint8_t*>(maskBits);

          mlir::Value statusRunning = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(0));
          mlir::Value statusFound   = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(1));
          mlir::Value statusFailed  = rewriter.create<mlir::arith::ConstantOp>(loc, i8Type, rewriter.getI8IntegerAttr(2));
          mlir::Value oneConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
          mlir::Value needleLenMinusOne  = rewriter.create<mlir::arith::ConstantIndexOp>(loc, needle.size() - 1);
          mlir::Value maxWhileIndex = rewriter.create<mlir::arith::SubIOp>(loc, indexType, endIdx, needleLenMinusOne);

          for (size_t j = 0; j < needle.size(); ++j) {
             needleBitsU8[j] = static_cast<uint8_t>(needle[j]);
             maskBitsU8[j] = 0xFF;
          }
          llvm::APInt maskAP(128, llvm::ArrayRef<uint64_t>(maskBits, 2));
          auto maskAttr =  rewriter.getIntegerAttr(i128Type, maskAP);
          mlir::Value maskVal = rewriter.create<mlir::arith::ConstantOp>(loc, i128Type, maskAttr);

          llvm::APInt needleAP(128, llvm::ArrayRef<uint64_t>(needleBits, 2));
          auto needleAttr =  rewriter.getIntegerAttr(i128Type, needleAP);
          mlir::Value needleVal= rewriter.create<mlir::arith::ConstantOp>(loc, i128Type, needleAttr);
          mlir::Value eightConst = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 8);

          auto loop = rewriter.create<mlir::scf::WhileOp>(loc, mlir::TypeRange{indexType, i8Type}, mlir::ValueRange{startIdx, statusRunning},
             [&](mlir::OpBuilder& opBuilder2, mlir::Location location2, mlir::ValueRange args) {
                mlir::Value running = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::eq, args[1], statusRunning);
                mlir::Value inRange = opBuilder2.create<mlir::arith::CmpIOp>(location2, mlir::arith::CmpIPredicate::ult, args[0], maxWhileIndex);
                opBuilder2.create<mlir::scf::ConditionOp>(location2, opBuilder2.create<mlir::arith::AndIOp>(location2, running, inRange), args);
             },

             [&](mlir::OpBuilder& opBuilder2, mlir::Location location2, mlir::ValueRange args) {
                mlir::Value offset = args[0];

                mlir::Value bitOffIdx = opBuilder2.create<mlir::arith::MulIOp>(location2, indexType, offset, eightConst);
                mlir::Value bitOff= opBuilder2.create<mlir::arith::IndexCastOp>(location2, i128Type, bitOffIdx);
                mlir::Value shifted = opBuilder2.create<mlir::arith::ShRUIOp>(location2, haystackInlined, bitOff);
                mlir::Value remaining = opBuilder2.create<mlir::arith::SubIOp>(location2, indexType, maxWhileIndex, offset);

                auto memchr = opBuilder2.create<util::InlineMemchr>(location2, i1Type, indexType, shifted, remaining, opBuilder2.getI8IntegerAttr(needle[0]));
                mlir::Value foundCurrent = memchr.getResult(0);
                mlir::Value hitAbs = opBuilder2.create<mlir::arith::AddIOp>(location2, indexType, offset, memchr.getResult(1));

                auto step = opBuilder2.create<mlir::scf::IfOp>(
                   location2,
                   foundCurrent,
                   // candidate at hitAbs: mask-compare the full needle
                   [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                      mlir::Value hitBitsIdx = opBuilder3.create<mlir::arith::MulIOp>(location3, indexType, hitAbs, eightConst);
                      mlir::Value hitBits = opBuilder3.create<mlir::arith::IndexCastOp>(location3, i128Type, hitBitsIdx);
                      mlir::Value window = opBuilder3.create<mlir::arith::ShRUIOp>(location3, haystackInlined, hitBits);
                      mlir::Value masked = opBuilder3.create<mlir::arith::AndIOp>(location3, window, maskVal);
                      mlir::Value isMatch = opBuilder3.create<mlir::arith::CmpIOp>(location3, mlir::arith::CmpIPredicate::eq, masked, needleVal);

                      mlir::Value nextOffset = opBuilder3.create<mlir::arith::AddIOp>(location3, indexType, hitAbs, oneConst);
                      mlir::Value next = opBuilder3.create<mlir::arith::SelectOp>(location3, isMatch, hitAbs, nextOffset);
                      mlir::Value nextStatus = opBuilder3.create<mlir::arith::SelectOp>(location3, isMatch, statusFound, statusRunning);
                      opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{next, nextStatus});
                   },
                   [&](mlir::OpBuilder& opBuilder3, mlir::Location location3) {
                      opBuilder3.create<mlir::scf::YieldOp>(location3, mlir::ValueRange{offset, statusFailed});
                   });

                opBuilder2.create<mlir::scf::YieldOp>(location2, step.getResults());
             });
          found = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, loop.getResult(1), statusFound);
          pos = loop.getResult(0);
      }
      rewriter.replaceOp(containsOp, mlir::ValueRange{found, pos});
      return mlir::success();
   }
};

class PrepareLowering : public mlir::PassWrapper<PrepareLowering, mlir::OperationPass<mlir::ModuleOp>> {
   virtual llvm::StringRef getArgument() const override { return "util-prepare-lowering"; }
   void getDependentDialects(mlir::DialectRegistry& registry) const override {
      registry.insert<mlir::scf::SCFDialect>();
      registry.insert<mlir::arith::ArithDialect>();   }

   public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareLowering)
   void runOnOperation() override {
      //transform "standalone" aggregation functions
      {
         mlir::RewritePatternSet patterns(&getContext());
         patterns.add<SplitConstLike>(&getContext());
         patterns.add<SpecializeStringContains>(&getContext());
         patterns.add<SpecializeBytesContains>(&getContext());
         if (lingodb::compiler::applyPatternsGreedily(getOperation().getRegion(), std::move(patterns)).failed()) {
            assert(false && "should not happen");
         }
      }

   }
};
} // namespace

std::unique_ptr<mlir::Pass> lingodb::compiler::dialect::util::createPrepareLoweringPass() { return std::make_unique<PrepareLowering>(); } // NOLINT(misc-use-internal-linkage)
