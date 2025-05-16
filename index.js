const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const csv = require("csv-parser");
const { URL } = require("url");

const urls = [];
const labels = [];

const SCAM_KEYWORDS = [
  "bantuankerajaan", "bantuanterkini", "madani", "bkmterkini", "bantuanrakyat",
  "privatevcs", "portalmykasih", "portalterkinimy", "portalkerajaan", "portalbantuan",
  "applymykad", "asasrahmah", "bantuanmadani", "kerajaan"
];

function calculateEntropy(str) {
  const len = str.length;
  const freqs = {};
  for (let char of str) {
    freqs[char] = (freqs[char] || 0) + 1;
  }
  let entropy = 0;
  for (let char in freqs) {
    const p = freqs[char] / len;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

function extractFeatures(url, flag = true) {
  url = url.includes("://") ? url : `http://${url}`;
  const urlObj = new URL(url);
  const hostname = urlObj.hostname;
  const pathname = urlObj.pathname;

  const tokens = url.toLowerCase()
    .replace(/https?:\/\//g, '')
    .split(/[\.\-\/\?\=\&\_]/)
    .filter(t => t.length > 0);

  const features = {
    dashCount: (hostname.match(/-/g) || []).length,
    digitCount: (hostname.match(/\d/g) || []).length,
    subdomainCount: hostname.split(".").length - 2,
    slashCount: (url.replace(/https?:\/\//g, '').match(/\//g) || []).length,
    hasIP: /^\d+\.\d+\.\d+\.\d+$/.test(hostname) ? 1 : 0,
    tldType: hostname.endsWith('.my') ? 0 : 1,
    entropy: calculateEntropy(hostname)
  };

  SCAM_KEYWORDS.forEach(keyword => {
    features[`kw_${keyword}`] = tokens.includes(keyword) ? 1 : 0;
  });

  return flag ? Object.values(features) : features;
}

function loadDataset(path) {
  return new Promise((resolve) => {
    fs.createReadStream(path)
      .pipe(csv())
      .on("data", (data) => {
        urls.push(extractFeatures(data.url));
        labels.push(Number(data.label));
      })
      .on("end", () => resolve());
  });
}

async function trainModel() {
  const xs = tf.tensor2d(urls);
  const ys = tf.tensor2d(labels, [labels.length, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [7 + SCAM_KEYWORDS.length], units: 12, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

  await model.fit(xs, ys, { epochs: 50, shuffle: true });

  await model.save("file://./model");
  console.log("Model trained and saved.");
}

async function predictURL(inputUrl) {
  const model = await tf.loadLayersModel("file://./model/model.json");
  const input = tf.tensor2d([extractFeatures(inputUrl)]);
  const prediction = model.predict(input);
  const result = (await prediction.data())[0];
  console.log(`Prediction for ${inputUrl}: ${result > 0.7 ? "Suspicious" : "Safe"} (${result.toFixed(2)})`);
  console.log(extractFeatures(inputUrl, false));
}

(async () => {
  async function train() {
    console.log("Training model...");
    await loadDataset("./data/urls.csv");
    await trainModel();
  }

  if (!fs.existsSync("./model/model.json")) {
    train();
  }

  const args = process.argv.slice(2);

  switch(args[0]) {
    case '--train':
      train();
      break;
    default:
      const inputUrl = args[0] || "https://app.mykasih.net/sara2/checkstatus";
      predictURL(inputUrl);
  }
})();
