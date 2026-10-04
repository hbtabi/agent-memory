/**
 * Minimal agent loop: recall → build prompt → (call your LLM) → remember.
 * Run: npm run build && node dist/examples/basic.js
 */
import { AgentMemory } from "../src/index.js";

const memory = await AgentMemory.open("examples/basic.memory.json");

memory.remember("User's name is Sam and they run a dog-walking business in Peckham.", { kind: "semantic" });
memory.remember("Sam is allergic to cats, never suggest cat-sitting work.", { kind: "semantic", pinned: true });
memory.remember("Sam asked for flyer copy for the spring dog-walking offer.", { tags: ["marketing"] });

const userMessage = "Can you write an Instagram post for my dog-walking business?";
const context = memory.buildContext(userMessage, { k: 3 });

const prompt = `${context}\n\nUser: ${userMessage}`;
console.log(prompt);

// const reply = await llm(prompt);                       // your model call here
memory.remember(`Sam asked: ${userMessage}`, { tags: ["marketing"] });
await memory.save();
console.log(`\n${memory.size} memories saved to examples/basic.memory.json`);
