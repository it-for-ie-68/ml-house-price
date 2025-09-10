import { useEffect, useState } from "react";
import { load_model, ort } from "./model";

function App() {
  const [output, setOutput] = useState<any>(null);
  const [areaText, setAreaText] = useState<string>("");
  const [session, setSession] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    setIsLoading(true);
    load_model().then((res) => {
      setSession(res);
      setIsLoading(false);
    });
  }, []);

  const prediction = handleOutput(output);

  function handleAreaText(e: any) {
    setAreaText(e.target.value);
  }

  async function handleSubmit() {
    const areaScaled = handleArea(areaText);
    if (!areaScaled || !session) return;
    // Prepare input (float32 tensor)
    const input = new ort.Tensor("float32", [areaScaled], [1, 1]);
    const feeds = { input: input };

    // Run inference
    const output = await session.run(feeds);
    // console.log({ output: output });
    setOutput(output);
  }

  if (isLoading) return <div className="container">Loading....</div>;
  return (
    <div className="container">
      <h1>House Price Prediction</h1>
      <label htmlFor="area">Area (Square Feet)</label>
      <input
        type="number"
        id="area"
        value={areaText}
        onChange={handleAreaText}
      />
      <button onClick={handleSubmit}>Submit</button>
      {/* Display output */}
      <div>
        {prediction && <article>Estimated Price: {prediction}💲</article>}
      </div>
    </div>
  );
}

export default App;

function handleOutput(output: any) {
  // Change here
  const X_min = 800000;
  const X_max = 2000000;
  //
  if (!output) return null;
  const X_scaled = output?.output?.cpuData["0"] as number;
  // console.log({ X_scaled });
  const X = X_min + X_scaled * (X_max - X_min);
  return X.toFixed(2);
}

function handleArea(input: string) {
  // Change here
  const mean = 1275.25;
  const std = 353.78480394;
  //
  const inputNum = parseFloat(input);
  if (isNaN(inputNum)) {
    return null;
  }
  return (inputNum - mean) / std;
}
