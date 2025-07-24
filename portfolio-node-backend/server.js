import express from "express";
import cors from "cors";
import fs from "fs/promises";
import path from "path";
import dotenv from "dotenv";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { Document } from "langchain/document";
import { Resend } from "resend";
import morgan from "morgan"

dotenv.config();
const app = express();
app.use(cors());
app.use(express.json());


app.use(morgan("dev"))

// Env config fallback (if not using .env file)
const HUGGINGFACE_API_KEY = process.env.KEY_HF
const GROQ_API_KEY = process.env.KEY_GROQ
const RESEND_API_KEY = process.env.KEY_RESEND

const resend = new Resend(RESEND_API_KEY);

// ✅ Email API (via Resend)
app.post("/send", async (req, res) => {
  const { name, email, message } = req.body;

  try {
    await resend.emails.send({
      from: "onboarding@resend.dev",
      to: "suraj442915@gmail.com",
      subject: `Message from ${name}`,
      text: `From: ${email}\n\n${message}`,
    });

    res.status(200).json({ success: true });
  } catch (error) {
    console.error("❌ Email Error:", error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// ✅ Chat + Embedding API
app.post("/chat", async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) return res.status(400).send("Missing prompt");

    const filePath = path.join(process.cwd(), "myfile.txt");
    const fileContent = await fs.readFile(filePath, "utf-8");
    const document = new Document({ pageContent: fileContent });

    const embeddings = new HuggingFaceInferenceEmbeddings({
      apiKey: HUGGINGFACE_API_KEY,
      model: "sentence-transformers/all-MiniLM-L6-v2",
    });

    const vectorStore = await MemoryVectorStore.fromDocuments([document], embeddings);
    const relevantDocs = await vectorStore.similaritySearch(prompt, 1);

    const model = new ChatGroq({
      model: "llama-3.3-70b-versatile",
      apiKey: GROQ_API_KEY,
    });

    const promptTemplate = ChatPromptTemplate.fromTemplate(
      `Answer the following question using the context provided. Use efficient and minimal words.
Context: {context}
Question: {prompt}`
    );

    const chain = await createStuffDocumentsChain({
      llm: model,
      prompt: promptTemplate,
    });

    const answer = await chain.invoke({
      context: relevantDocs,
      prompt,
    });

    res.send(answer);
  } catch (err) {
    console.error("❌ API Error:", err);
    res.status(500).send("Internal Server Error");
  }
});

// ✅ Run Server I don't have domain so resolve resend logic so I can't use it don't change other logic resolve error mail not coming

app.listen(4000, () => {
  console.log("🚀 Server running at http://localhost:4000");
});
