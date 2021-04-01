<script>
  import * as tf from "@tensorflow/tfjs";
  import * as use from "@tensorflow-models/universal-sentence-encoder";
  import seedrandom_prng from "seedrandom";
  //import * as knn from "@tensorflow-models/knn-classifier";
  import Category from "./Category.svelte";
  import { categories, classifier, layers, predictions } from "./stores";
  import { pullSampleData, convertToTfDataset, getTopKClasses } from "./utils";

  import "./mystyles.scss";

  let trainingMessage = "Train Model";
  let isTraining = false;

  let hasTrained = false;
  let isPredicting = false;
  
  let pred = "";
  let epochs = 50;
  let learningRate = 0.001;
  let batchSize = 16;

  function addCategory({ label = "New Category", training = "" } = {}) {
    //console.log($categories);
    let newId;
    if ($categories.length > 0) {
      newId = Math.max(...$categories.map((category) => category.id)) + 1;
    } else {
      newId = 1;
    }

    const newCategory = {
      id: newId,
      label,
      training,
      loading: false,
      trainingTensors: [],
    };

    categories.update((all) => [...all, newCategory]);
  }

  function removeCategory(id) {
    // this method works as long as category IDs are unique. It should return all categories except one of them
    categories.update((all) => all.filter((category) => category.id !== id));
  }

  // takes a line-break delimited set of sentences and converts them into vectors using the universal sentence encoder
  async function encode(sentences) {
    const model = await use.load();
    const embeddings = await model.embed(sentences.split("\n"));
    return embeddings;
  }
  async function runTraining() {
    isTraining = true;
    document.body.style.cursor = "wait";
    trainingMessage = "Preparing Training Examples...";
    await Promise.all(preTrain());
    await train();
    isTraining = false;
    hasTrained = true;
    document.body.style.cursor = "";
  }
  function preTrain() {
    return $categories.map((category) =>
      setExamples(category.training, category.id)
    );
  }

  async function train() {
    const callbacks = {
      onTrainEnd: () => {
        trainingMessage = "Retrain Model";
        console.log("Training Complete!");
      },
      onEpochBegin: (epoch) => {
        trainingMessage = `Training cycle ${epoch}/${epochs}`;
      },
      onEpochEnd: (epoch) => console.log("Done with epoch", epoch),
    };

    const allTrainingTensors = $categories.map((category) => {
      if (category.trainingTensors.length < 1) {
        throw new Error(
          `Category "${category.label}" has no training examples!`
        );
      }
      return category.trainingTensors;
    });

    const numClasses = $categories.length;

    if (numClasses < 2) {
      throw new Error(`You need to add more categories`);
    }

    const { trainDataset, validationDataset } = convertToTfDataset(
      allTrainingTensors,
      numClasses,
      seedrandom_prng("kodama")
    );

    const inputShape = [1, 512];
    const inputSize = tf.util.sizeFromShape(inputShape);

    const varianceScaling = tf.initializers.varianceScaling({ seed: 3.14 });

    const customLayers = [];
    $layers.forEach((layer) => {
      if (customLayers.length == 0) {
        customLayers.push(
          tf.layers.dense({
            inputShape: [inputSize],
            units: layer.units,
            activation: layer.activation ?? "relu",
            kernelInitializer: varianceScaling, // 'varianceScaling'
            useBias: true,
          })
        );
      } else {
        customLayers.push(
          tf.layers.dense({
            units: layer.units,
            activation: layer.activation ?? "relu",
            kernelInitializer: varianceScaling, // 'varianceScaling'
            useBias: true,
          })
        );
      }
    });
    //console.log(customLayers);
    const trainingModel = tf.sequential({
      layers: [
        ...customLayers,
        tf.layers.dense({
          kernelInitializer: varianceScaling, // 'varianceScaling'
          useBias: false,
          activation: "softmax",
          units: numClasses,
        }),
      ],
    });

    const optimizer = tf.train.adam(learningRate);
    // const optimizer = tf.train.rmsprop(params.learningRate);

    trainingModel.compile({
      optimizer,
      // loss: 'binaryCrossentropy',
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    if (!(batchSize > 0)) {
      throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction`
      );
    }

    const trainData = trainDataset.batch(batchSize);
    const validationData = validationDataset.batch(batchSize);

    const history = await trainingModel.fitDataset(trainData, {
      epochs: epochs,
      validationData,
      callbacks,
    });

    classifier.set(trainingModel);
  }

  async function setExamples(sentences, id) {
    const embeddings = await encode(sentences);
    const newTrainingTensors = [];
    tf.split(embeddings, embeddings.shape[0], 0).forEach((tensor) => {
      newTrainingTensors.push(tensor);
    });
    categories.update((categories) => {
      const categoryIndex = categories.findIndex((item) => item.id == id);
      categories[categoryIndex].trainingTensors = newTrainingTensors;
      return categories;
    });
  }

  async function predict(sentence) {
    isPredicting = true;
    const predEncoded = await encode(sentence);
    const logits = await $classifier.predict(predEncoded);
    const labels = $categories.map((category) => category.label);
    const results = await getTopKClasses(labels, logits);

    console.log(results);
    isPredicting = false;
    return { sentence, results };

    //console.log(await $classifier.predictClass(predEncoded));
  }

  async function loadExample() {
    const positive = await pullSampleData("./samples/positive.txt");
    const negative = await pullSampleData("./samples/negative.txt");
    const neutral = await pullSampleData("./samples/neutral.txt");
    categories.set([]);
    addCategory({ label: "Positive", training: positive });
    addCategory({ label: "Negative", training: negative });
    addCategory({ label: "Neutral", training: neutral });
  }

  function removeLayer(id) {
    layers.update((all) => all.filter((layer) => layer.id !== id));
  }

  function addLayer({ units = 100 } = {}) {
    let newId;
    if ($layers.length > 0) {
      newId = Math.max(...$layers.map((layer) => layer.id)) + 1;
    } else {
      newId = 1;
    }
    const newLayer = {
      id: newId,
      units,
    };
    layers.update((layers) => [...layers, newLayer]);
  }
</script>

<main>
  <section class="hero is-danger">
    <div class="hero-body">
      <p class="title">Teachable Text Machine</p>
    </div>
  </section>
  <section class="section">
    <nav class="level">
      <div class="level-left">
        <div class="level-item">
          <div class="field">
            <div class="control">
              <button
                class="button"
                on:click={addCategory}
                disabled={isTraining}>Add Category</button
              >
            </div>
          </div>
        </div>
        <div class="level-item">
          <div class="field">
            <div class="control">
              <button
                class="button"
                on:click={loadExample}
                disabled={isTraining}>Use Example Data (twitter)</button
              >
            </div>
          </div>
        </div>
        <div class="level-item">
          <div class="field">
            <div class="control">
              <button
                class="button is-success"
                on:click={runTraining}
                disabled={isTraining}>{trainingMessage}</button
              >
            </div>
          </div>
        </div>
      </div>
    </nav>
  </section>

  <div class="columns is-desktop">
    <div class="column">
      {#each $categories as category (category.id)}
        <Category
          id={category.id}
          bind:label={category.label}
          bind:training={category.training}
          remove={() => removeCategory(category.id)}
          loading={isTraining}
        />
      {/each}
    </div>
    <div class="column is-narrow">
      <div class="box">
        <h4 class="title is-4">Model Settings</h4>
        <div class="field">
          <div class="control">
            <button class="button" on:click={addLayer} disabled={isTraining}
              >Add Layer</button
            >
          </div>
        </div>

        {#each $layers as layer (layer.id)}
          <div class="field is-horizontal">
            <div class="field-label is-normal">
              <label class="label"># Units: </label>
            </div>
            <div class="field-body">
              <div class="field">
                <div class="control">
                  <input
                    type="number"
                    class="input"
                    min="10"
                    bind:value={layer.units}
                    disabled={isTraining}
                  />
                </div>
              </div>
              <div class="field">
                <div class="control">
                  <button
                    class="button"
                    on:click={() => removeLayer(layer.id)}
                    disabled={isTraining || $layers.length == 1}
                    >Delete Layer</button
                  >
                </div>
              </div>
            </div>
          </div>
        {/each}
      </div>
      <div class="box">
        <h4 class="title is-4">Training Settings</h4>
        <div class="field">
          <label class="label">Training Cycles</label>
          <div class="control">
            <input
              class="input"
              type="number"
              min="1"
              bind:value={epochs}
              disabled={isTraining}
            />
          </div>
        </div>

        <div class="field">
          <label class="label">Training Size</label>
          <div class="control">
            <input
              class="input"
              type="number"
              min="1"
              bind:value={batchSize}
              disabled={isTraining}
            />
          </div>
        </div>

        <div class="field">
          <label class="label">Learning Rate </label>
          <div class="control">
            <input
              class="input"
              type="number"
              step="0.0001"
              bind:value={learningRate}
              disabled={isTraining}
            />
          </div>
        </div>
      </div>
    </div>

    <div class="column">
      <div class="box">
        <h4 class="title is-4">Evaluate Model</h4>
        <div class="field">
          <div class="control">
            <input
              type="text"
              class="input"
              bind:value={pred}
              disabled={isTraining || !hasTrained || isPredicting}
            />
          </div>
        </div>

        <div class="field">
          <div class="control">
            <button
              class="button"
              on:click={() =>
                predict(pred).then((output) =>
                  predictions.update((allPreds) => [output, ...allPreds])
                )}
              disabled={isTraining || !hasTrained || isPredicting}>{isPredicting? "Predicting..." : "Predict"}</button
            >
          </div>
        </div>
      </div>
      {#each $predictions as prediction}
        <div class="box">
          <h6 class="title is-6">{prediction.sentence}</h6>
          <table class="table">
            <thead>
              <tr>
                <th>Category</th>
                <th>Probability</th>
              </tr>
            </thead>
            <tbody>
              {#each prediction.results as category}
                <tr class={category.className === prediction.results[0].className ? 'is-selected' : ''} >
                  <th>{category.className}</th>
                  <th>{(category.probability * 100).toFixed(4)}%</th>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/each}
    </div>
  </div>
</main>
