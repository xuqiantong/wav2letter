/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common/FlashlightUtils.h"
#include "ContentAttention.h"
#include <cmath>
#include <iostream>

using namespace fl;

namespace w2l {

Variable attentionMask(
    const Variable& input,
    const af::array& mask) {
  af::array output = af::flat(input.array());
  af::array flatMask = af::flat(mask);
  auto inputDims = input.dims();

  output(flatMask) = -100000;
  output = af::moddims(output, inputDims);
  
  auto gradFunc =
      [flatMask, inputDims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        af::array gradArray = af::flat(gradOutput.array());
        gradArray(flatMask) = 0.;
        auto grad = Variable(af::moddims(gradArray, inputDims), false);
        inputs[0].addGrad(grad);
      };
  return Variable(output, {input.withoutData()}, gradFunc);
}

std::pair<Variable, Variable> ContentAttention::forward(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& /* unused */,
    const Variable& attnWeight,
    const Variable& inputProportions) {
  int dim = xEncoded.dims(0);
  if (dim != (1 + keyValue_) * state.dims(0)) {
    throw std::invalid_argument("Invalid dimension for content attention");
  }

  auto keys = keyValue_ ? xEncoded(af::seq(0, dim / 2 - 1)) : xEncoded;
  auto values = keyValue_ ? xEncoded(af::seq(dim / 2, dim - 1)) : xEncoded;

  // [targetlen, seqlen, batchsize]
  auto innerProd = matmulTN(state, keys) / std::sqrt(state.dims(0));

  if (!attnWeight.isempty()) {
    innerProd = innerProd + attnWeight;
  }
  af::array padMask;
  if (FLAGS_attention_mask && !inputProportions.isempty()) {
    int B = innerProd.dims(2);
    int T = innerProd.dims(1);
    af::array inputNotPaddedSize = af::moddims(af::ceil(inputProportions.array() * T), af::dim4(1, B)); 
    padMask = af::iota(af::dim4(T, 1), af::dim4(1, B)) >= af::tile(inputNotPaddedSize, T, 1); 
    padMask = af::tile(af::moddims(padMask, af::dim4(1, T, B)), innerProd.dims(0), 1, 1);
    innerProd = attentionMask(innerProd, padMask);
  }
  // [targetlen, seqlen, batchsize]
  auto attention = softmax(innerProd, 1);
  // [hiddendim, targetlen, batchsize]
  auto summaries = matmulNT(values, attention);
  return std::make_pair(attention, summaries);
}

std::string ContentAttention::prettyString() const {
  return "ContentBasedAttention";
}

NeuralContentAttention::NeuralContentAttention(int dim, int layers /* = 1 */) {
  Sequential net;
  net.add(ReLU());
  for (int i = 1; i < layers; i++) {
    net.add(Linear(dim, dim));
    net.add(ReLU());
  }
  net.add(Linear(dim, 1));
  add(net);
}

std::pair<Variable, Variable> NeuralContentAttention::forward(
    const Variable& state,
    const Variable& xEncoded,
    const Variable& /* unused */,
    const Variable& attnWeight,
    const Variable& ) {
  int U = state.dims(1);
  int H = xEncoded.dims(0);
  int T = xEncoded.dims(1);
  int B = xEncoded.dims(2);

  auto tileHx = tile(moddims(xEncoded, {H, 1, T, B}), {1, U, 1, 1});
  auto tileHy = tile(moddims(state, {H, U, 1, B}), {1, 1, T, 1});

  // [hiddendim, targetlen, seqlen, batchsize]
  auto hidden = tileHx + tileHy;

  // [targetlen, seqlen, batchsize]
  auto nnOut = moddims(module(0)->forward({hidden}).front(), {U, T, B});

  if (!attnWeight.isempty()) {
    nnOut = nnOut + attnWeight;
  }

  // [targetlen, seqlen, batchsize]
  auto attention = softmax(nnOut, 1);

  // [hiddendim, targetlen, batchsize]
  auto summaries = matmulNT(xEncoded, attention);

  return std::make_pair(attention, summaries);
}

std::string NeuralContentAttention::prettyString() const {
  return "NeuralContentBasedAttention";
}

} // namespace w2l
