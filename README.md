# P2P Federated Learning with Tensorflow and libp2p

This example shows how to train MNIST with FedAvg algorithm using libp2p nodes. The repo will be updated regularly. All contributions are welcome.

Prepare the node environment:
```sh
$ npm install
# Or
$ yarn
```

Run the training script in seperated terminal:
```sh
$ node main.js <node_id>
```
There are three nodes currently with their config. You can add your config in ```data/nodes``` and increase node number to measure scalability of the system.