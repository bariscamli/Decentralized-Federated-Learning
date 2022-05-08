/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const tf = require('@tensorflow/tfjs');

const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 6,
  kernelSize: 5,
  strides:1,
  activation: 'tanh',
}));
model.add(tf.layers.averagePooling2d({poolSize: [2, 2]}));
model.add(tf.layers.conv2d({
  filters: 16,
  kernelSize: 5,
  strides:1,
  activation: 'tanh',
}));
model.add(tf.layers.averagePooling2d({poolSize: [2, 2]}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({units: 120, activation: 'tanh'}));
model.add(tf.layers.dense({units: 84, activation: 'tanh'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

model.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

module.exports = model;
