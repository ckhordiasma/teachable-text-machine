<script>
  import * as tf from "@tensorflow/tfjs";
  import * as use from "@tensorflow-models/universal-sentence-encoder";
  import * as knn from "@tensorflow-models/knn-classifier";
  let cat1 = "";
  let cat2 = "";
  let pred = '';
  const training = {};
  const classifier = knn.create();

  async function encode(sentences){
	  const model = await use.load();
	  const embeddings = await model.embed(sentences.split("\n"));
	  return embeddings;
  }

  async function addCategory(sentences, category) {
    
    training[category] = await encode(sentences);
    
  }

  async function train() {
    let i = 0;
    let examples;
    examples = training["1"];
    tf.split(examples, examples.shape[0], 0).forEach((tensor) => {
      classifier.addExample(tensor, 1);
    });

    examples = training["2"];
    tf.split(examples, examples.shape[0], 0).forEach((tensor) => {
      classifier.addExample(tensor, 2);
    });
  }

  async function predict(){
	const predEncoded = await encode(pred);
	console.log(await classifier.predictClass(predEncoded));
  }
</script>

<main>
  <textarea bind:value={cat1} />

  <button on:click={() => addCategory(cat1, 1)}> Add Category </button>
  <textarea bind:value={cat2} />
  <button on:click={() => addCategory(cat2, 2)}> Add Category </button>

  <button on:click={train}> Training</button>

  <input type="text" bind:value={pred} />
  <button on:click={predict}> Predict</button>
</main>

<style>
  main {
    text-align: center;
    padding: 1em;
    max-width: 240px;
    margin: 0 auto;
  }

  h1 {
    color: #ff3e00;
    text-transform: uppercase;
    font-size: 4em;
    font-weight: 100;
  }

  @media (min-width: 640px) {
    main {
      max-width: none;
    }
  }
</style>
