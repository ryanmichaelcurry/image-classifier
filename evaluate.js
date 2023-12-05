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

var NAMES = [];

async function readAndPreprocessImage(imagePath) {
    try {
        const imageBuffer = fs.readFileSync(imagePath);

        const buffer = await sharp(imageBuffer)
            .resize(IMAGE_WIDTH, IMAGE_HEIGHT)
            .ensureAlpha() // We default to having alpha channels
            .png()
            .toBuffer();

        NAMES.push(imagePath);

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

async function loadTestData() {
    const testImages = [];
    const testLabels = [];

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

    const rows = []; // Array to store CSV row promises
    return new Promise((resolve, reject) => {
        fs.createReadStream(`${DIR}/evaluate/images.csv`)
            .pipe(csv())
            .on('data', (data) => {
                // Push a new promise for each row into the rows array
                rows.push((async () => {
                    var imagePathPng = `${DIR}/evaluate/images/${data.Name}.png`;
                    var imagePathJpg = `${DIR}/evaluate/images/${data.Name}.jpg`;

                    var imageTensor;
                    if (fs.existsSync(imagePathPng)) {
                        imageTensor = await readAndPreprocessImage(imagePathPng);
                    } else if (fs.existsSync(imagePathJpg)) {
                        imageTensor = await readAndPreprocessImage(imagePathJpg);
                    } else {
                        throw new Error(`Image file not found for ${data.Name}`);
                    }

                    // ChatGPT :)
                    const types = Object.keys(data)
                                        .filter(key => key.startsWith('Type'))
                                        .map(key => data[key])
                                        .filter(type => type);

                    const labelArray = encodeTypesToOneHot(types);

                    if (!Array.isArray(labelArray)) {
                        throw new Error(`Label encoding did not return an array for ${data.Name}`);
                    }
  
                    const labelTensor = tf.tensor1d(labelArray);

                    testImages.push(imageTensor);
                    testLabels.push(labelTensor);
                })());
            })
            .on('end', async () => {
                await Promise.all(rows);

                resolve({
                    testData: tf.stack(testImages),
                    testLabels: tf.stack(testLabels)
                });

            });
    });
}

async function evaluateModel(model, testData, testLabels) {
    // I could not figure out how to display the prediction, so the predictions.array() code below is ChatGPT
    const predictions = model.predict(testData);
    predictions.array().then(predArrays => {
        testLabels.array().then(labelArrays => {
            for (let i = 0; i < predArrays.length; i++) {
                const predIndex = predArrays[i].indexOf(Math.max(...predArrays[i]));
                const labelIndex = labelArrays[i].indexOf(1);

                if (predIndex !== labelIndex) {
                    console.log(`Mismatch on image ${NAMES[i]}: Predicted: ${TYPES[predIndex]}, Actual: ${TYPES[labelIndex]}`);
                }
            }
        });
    });

    const evaluation = model.evaluate(testData, testLabels);
    const [testLoss, testAccuracy] = await Promise.all(evaluation.map(t => t.data()));

    console.log(`Test Loss: ${testLoss[0]}`);
    console.log(`Test Accuracy: ${testAccuracy[0]}`);
}

async function main() {
    // Load the trained model
    const model = await tf.loadLayersModel(`file://${DIR}/models/${DIR}${EPOCHS}/model.json`);

    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Load the test data
    const { testData, testLabels } = await loadTestData();

    // Evaluate the model
    await evaluateModel(model, testData, testLabels);
}

main();