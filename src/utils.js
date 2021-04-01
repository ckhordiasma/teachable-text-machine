import * as tf from "@tensorflow/tfjs";

const VALIDATION_FRACTION = 0.15;

// extracts sample data from a file, limits to the first 500 samples
export async function pullSampleData(url){
    const response = await fetch(url);
    const text = await response.text();

    return fisherYates(text.split('\n')).slice(0,100).join('\n');  
  }

/**
 * Shuffle an array of Float32Array or Samples using Fisher-Yates algorithm
 * Takes an optional seed value to make shuffling predictable
 */
 export function fisherYates(array, seed) {
    const length = array.length;

    // need to clone array or we'd be editing original as we goo
    const shuffled = array.slice();

    for (let i = (length - 1); i > 0; i -= 1) {
        let randomIndex ;
        if (seed) {
            randomIndex = Math.floor(seed() * (i + 1));
        }
        else {
            randomIndex = Math.floor(Math.random() * (i + 1));
        }

        [shuffled[i], shuffled[randomIndex]] = [shuffled[randomIndex],shuffled[i]];
    }

    return shuffled;
}

/**
 * Converts an integer into its one-hot representation and returns
 * the data as a JS Array.
 */
 function flatOneHot(label, numClasses) {
    const labelOneHot = new Array(numClasses).fill(0);
    labelOneHot[label] = 1;

    return labelOneHot;
}

    /**
     * Process the examples by first shuffling randomly per class, then adding
     * one-hot labels, then splitting into training/validation datsets, and finally
     * sorting one last time
     */
// examples is an array of array of tensors
export function convertToTfDataset(examples, numClasses, seed){
// first shuffle each class individually
        // TODO: we could basically replicate this by insterting randomly
        for (let i = 0; i < examples.length; i++) {
            examples[i] = fisherYates(examples[i], seed);
        }

        // then break into validation and test datasets

        let trainDataset = [];
        let validationDataset = [];

        // for each class, add samples to train and validation dataset
        for (let i = 0; i < examples.length; i++) {
            const y = flatOneHot(i, numClasses);

            const classLength = examples[i].length;
            const numValidation = Math.ceil(VALIDATION_FRACTION * classLength);
            const numTrain = classLength - numValidation;

            const classTrain = examples[i].slice(0, numTrain).map((dataArray) => {
                return { data: dataArray, label: y };
            });

            const classValidation = examples[i].slice(numTrain).map((dataArray) => {
                return { data: dataArray, label: y };
            });

            trainDataset = trainDataset.concat(classTrain);
            validationDataset = validationDataset.concat(classValidation);
        }

        // finally shuffle both train and validation datasets
        trainDataset = fisherYates(trainDataset, seed) ;
        validationDataset = fisherYates(validationDataset, seed);
        
        const trainX = tf.data.array(trainDataset.map(sample => sample.data));
        const validationX = tf.data.array(validationDataset.map(sample => sample.data));
        const trainY = tf.data.array(trainDataset.map(sample => sample.label));
        const validationY = tf.data.array(validationDataset.map(sample => sample.label));

        // return tf.data dataset objects
        return {
            trainDataset: tf.data.zip({ xs: trainX,  ys: trainY}),
            validationDataset: tf.data.zip({ xs: validationX,  ys: validationY})
        };
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
 export async function getTopKClasses(labels, logits, topK = 3) {
    const values = await logits.data();
    return tf.tidy(() => {
        topK = Math.min(topK, values.length);
  
        const valuesAndIndices = [];
        for (let i = 0; i < values.length; i++) {
            valuesAndIndices.push({value: values[i], index: i});
        }
        valuesAndIndices.sort((a, b) => {
            return b.value - a.value;
        });
        const topkValues = new Float32Array(topK);
        const topkIndices = new Int32Array(topK);
        for (let i = 0; i < topK; i++) {
            topkValues[i] = valuesAndIndices[i].value;
            topkIndices[i] = valuesAndIndices[i].index;
        }
  
        const topClassesAndProbs = [];
        for (let i = 0; i < topkIndices.length; i++) {
            topClassesAndProbs.push({
                className: labels[topkIndices[i]], //IMAGENET_CLASSES[topkIndices[i]],
                probability: topkValues[i]
            });
        }
        return topClassesAndProbs;
    });
  }