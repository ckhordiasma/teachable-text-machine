<script>
  export let training;

  export let label;
  export let remove;
  export let id;
  export let loading;
 $: actualSamples = training.trim() == ''? 0: training.trim().split('\n').length;
</script>

<main class="box">
  
    <div class="field">
      <label class="label" for="categoryName{id}"
        >Category Name ( ID: {id} )</label
      >
      <div class="control">
        <input
          id="categoryName{id}"
          type="text"
          class="input"
          disabled={loading}
          bind:value={label}
        />
      </div>
    </div>
    <div class="field">
      <label class="label" for="training{id}">Training Sentences ({actualSamples})</label>
      <div class="control">
        <textarea
          id="training{id}"
          class="textarea"
          placeholder="Enter training here, one sample per line"
          disabled={loading}
          bind:value={training}
        />
      </div>
    </div>
    <div class="field">
      <p class="control">
        <button class="button" on:click={remove} disabled={loading}> Remove category </button>
      </p>
    </div>
</main>
