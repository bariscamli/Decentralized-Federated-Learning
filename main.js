/* eslint-disable no-console */
'use strict'

const PeerId = require('peer-id')
const { Multiaddr } = require('multiaddr')
const Libp2p = require('libp2p')
const TCP = require('libp2p-tcp')
const Mplex = require('libp2p-mplex')
const {NOISE}  = require('libp2p-noise')
const Gossipsub = require('libp2p-gossipsub')
const nodeAddressPort = require('./data/nodes/address.json')

const currentNodeId = process.argv.slice(2)[0]

const tf = require('@tensorflow/tfjs-node')
//const data = require('./data')(currentNodeId)
const data = require('./data')(1)
const model = require('./model')
const utils = require('./utils')
const COMMUNICATION_ROUND = 5

const createNode = async (peerAddress, peerIdFromJson) => {
  const node = await Libp2p.create({
    peerId: peerIdFromJson,
    addresses: {
      listen: [peerAddress]
    },
    modules: {
      transport: [TCP],
      streamMuxer: [Mplex],
      connEncryption: [NOISE],
      pubsub: Gossipsub
    }
  })
  await node.start()
  return node
}

  ; (async () => {
    await data.loadData()
    const { images: trainImages, labels: trainLabels } = data.getTrainData()
    const { images: testImages, labels: testLabels } = data.getTestData()

    const topic = 'mnist'
    let currentRound = 0

    const nodesArray = await Promise.all([
      PeerId.createFromJSON(require('./data/nodes/peer-id-node-1')),
      PeerId.createFromJSON(require('./data/nodes/peer-id-node-2')),
      PeerId.createFromJSON(require('./data/nodes/peer-id-node-3')),
    ])

    const nodes = {}
    nodesArray.forEach((v, i) => nodes[i] = v)

    const nodeAddress = nodes[(parseInt(currentNodeId) - 1).toString()]
    delete nodes[(parseInt(currentNodeId) - 1).toString()]

    const node = await createNode(nodeAddressPort[currentNodeId], nodeAddress)

    console.log("Current Node Address: ")
    node.multiaddrs.forEach((ma) => {
      console.log(ma.toString())
    })

    while (Object.keys(nodes).length > 0) {
      await utils.sleep(2000)
      for (const [key, value] of Object.entries(nodes)) {
        try {
          await node.dial(new Multiaddr(`${nodeAddressPort[(parseInt(key) + 1).toString()]}/p2p/${value.toB58String()}`))
          console.log("Connected !")
          delete nodes[key]
        } catch (AggregateError) {
          console.log("Connection Error")
        }
      }
    }

    console.log("All Nodes Connected !")

    const globalModel = await utils.modelToDict(model)
    const recievedNode = []

    node.pubsub.on(topic, (msg) => {
      const receivedModel = JSON.parse(msg.data)
      console.log("Model arrived from: ", receivedModel["nodeId"])

      if (receivedModel["roundIndex"] == currentRound) {
        recievedNode.push(receivedModel["nodeId"])
        delete receivedModel["roundIndex"]
        delete receivedModel["nodeId"]
        if (Object.keys(recievedNode).length == 1) {
          for (const layerName of Object.keys(receivedModel)) {
            globalModel[layerName] = receivedModel[layerName]
          }
        }
        else {
          for (const layerName of Object.keys(receivedModel)) {
            // weight
            utils.dictSum(globalModel[layerName]["data"][0], receivedModel[layerName]["data"][0])
            //bias
            utils.dictSum(globalModel[layerName]["data"][1], receivedModel[layerName]["data"][1])
          }
        }
      }
    })

    await node.pubsub.subscribe(topic)

    while (COMMUNICATION_ROUND > currentRound) {
      console.log("Current Round:", currentRound + 1)

      await model.fit(trainImages, trainLabels, {
        epochs: 1,
        batchSize: 10
      })
      console.log("Train End!")
      const evalOutput = model.evaluate(testImages, testLabels)
      console.log(
        `\nEvaluation result:\n` +
        `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; ` +
        `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`)


      const trainedModelToDict = await utils.modelToDict(model)
      trainedModelToDict["roundIndex"] = currentRound
      trainedModelToDict["nodeId"] = currentNodeId
      const serializedArray = JSON.stringify(trainedModelToDict)
      await node.pubsub.publish(topic, serializedArray)
      console.log("Model Published!!")

      while (Object.keys(recievedNode).length < 2) {await utils.delay(1 * 1000)}

      for (const layerName of Object.keys(globalModel)) {
        // weight
        utils.dictSum(globalModel[layerName]["data"][0], trainedModelToDict[layerName]["data"][0])
        utils.dictDivide(globalModel[layerName]["data"][0], 3)
        // bias
        utils.dictSum(globalModel[layerName]["data"][1], trainedModelToDict[layerName]["data"][1])
        utils.dictDivide(globalModel[layerName]["data"][1], 3)
      }

      utils.dictToModel(model, globalModel)
      recievedNode.length = 0
      currentRound += 1
    }

  })()
