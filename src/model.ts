import * as ort from "onnxruntime-web";
export async function load_model() {
  // Path must be relative to public folder in Vite projects
  const modelUrl = `/model.onnx`;

  // Create session
  const session = await ort.InferenceSession.create(modelUrl);
  return session;
}

export { ort };
