/*
    Author: Ryan Michael Curry
    Professor: Dr. Eric Hansen
    Class: Intro to Artificial Intelligence
    Date: December 5th, 2023
*/

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const csv = require('csv-parser');
const sharp = require('sharp');

const IMAGE_WIDTH = 120;
const IMAGE_HEIGHT = 120;
const NUM_CHANNELS = 4; // RGBA channels
var TYPES = [];

var DIR = "pokemon";
var EPOCHS = 20;

async function readAndPreprocessImage(imagePath) {
    try {
        const imageBuffer = fs.readFileSync(imagePath);

        const buffer = await sharp(imageBuffer)
            .resize(IMAGE_WIDTH, IMAGE_HEIGHT)
            .ensureAlpha() // We default to having alpha channels
            .png()
            .toBuffer();

        // Decode the image buffer into a tensor with 4 channels (RGBA)
        return tf.node.decodeImage(buffer, NUM_CHANNELS).toFloat().div(tf.scalar(255));

    } catch (error) {
        console.error(`Error processing image at path ${imagePath}: ${error.message}`);
        throw error;
    }
}

function encodeTypesToOneHot(types) {
    let oneHot = new Array(TYPES.length).fill(0);

    types.forEach(type => {
        if (type && TYPES.includes(type)) {
            oneHot[TYPES.indexOf(type)] = 1;
        }
    });

    return oneHot;
}

async function loadImageForEntry(data) {
    try {
        var imagePathPng = `${DIR}/train/images/${data.Name}.png`;
        var imagePathJpg = `${DIR}/train/images/${data.Name}.jpg`;

        let image;
        if (fs.existsSync(imagePathPng)) {
            image = await readAndPreprocessImage(imagePathPng);
        } else if (fs.existsSync(imagePathJpg)) {
            image = await readAndPreprocessImage(imagePathJpg);
        } else {
            throw new Error(`Image file not found for ${data.Name}`);
        }

        // ChatGPT :)
        const types = Object.keys(data)
                           .filter(key => key.startsWith('Type'))
                           .map(key => data[key])
                           .filter(type => type);

        return {
            image: image,
            label: encodeTypesToOneHot(types)
        };

    } catch (error) {
        console.error(`Error processing image for ${data.Name}: ${error}`);
        throw error;
    }
}

async function loadData() {
    const typeSet = new Set();

    // First pass to collect types
    await new Promise((resolve, reject) => {
        fs.createReadStream(`${DIR}/train/images.csv`)
            .pipe(csv())
            .on('data', (data) => {
                Object.keys(data).forEach(key => {
                    if (key.startsWith('Type')) {
                        typeSet.add(data[key]);
                    }
                });
            })
            .on('end', resolve);
    });

    TYPES = Array.from(typeSet);

    // Create Training Data
    const entries = [];
    return new Promise((resolve, reject) => {
        fs.createReadStream(`${DIR}/train/images.csv`)
            .pipe(csv())
            .on('data', (data) => {
                entries.push(data);
            })
            .on('end', async () => {
                try {
                    const processedEntries = await Promise.all(entries.map(e => loadImageForEntry(e)));
                    const images = processedEntries.map(e => e.image);
                    const labels = processedEntries.map(e => e.label);

                    resolve({
                        trainingData: tf.stack(images),
                        trainingLabels: tf.stack(labels),
                    });
                } catch (error) {
                    reject(error);
                }
            });
    });
}

function createModel() {
    const model = tf.sequential();

    // Convolutional layer 1
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS],
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));

    // Max-Pooling 1
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    // Convolutional layer 2
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
        padding: 'same'
    }));

    // Max-Pooling 2
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    // Flatten
    model.add(tf.layers.flatten());

    // Dense layer
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    // Output layer
    model.add(tf.layers.dense({ units: TYPES.length, activation: 'softmax' }));

    return model;
}

async function trainModel(model, trainingData, trainingLabels) {
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return await model.fit(trainingData, trainingLabels, {
        epochs: EPOCHS,
        }).then(info => {
            console.log('Final accuracy', info.history.acc);
    });
}

async function main() {
    const { trainingData, trainingLabels } = await loadData();

    const model = createModel();

    await trainModel(model, trainingData, trainingLabels);

    await model.save(`file://${DIR}/models/'${DIR}${EPOCHS}`);
}

main();