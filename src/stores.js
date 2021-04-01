import { writable } from "svelte/store";

export const categories = writable([
  { id: 1, label: "Positive", training: "", loading: false, trainingTensors: [] },
  { id: 2, label: "Negative", training: "", loading: false, trainingTensors: [] },
]);

export const layers = writable([{id: 1, units:200}]);
export const predictions = writable([]);
export const classifier = writable({});
