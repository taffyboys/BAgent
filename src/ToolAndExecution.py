#!/usr/bin/env python3

import json
import os
from pydoc import describe
import subprocess

from openai.auth import WorkloadIdentity

try:
    import readline
    # macOS 的 libedit 在处理中文输入时有退格问题，这四行修复它
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)
WORKDIR = Path.cwd()

client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

SYSTEM = f"You are a coding agent at {WORKDIR}. Use bash to solve tasks. Act, don't explain."

# ── Tool execution ────────────────────────────────────────
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path: str, limit: int | None = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

# ── Tool definition────────────────────────────
TOOLS = [
    {"type": "function","function": {
     "name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
     "name": "read_file", "description": "Read file contents.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {
     "name": "write_file", "description": "Write content to a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {
     "name": "edit_file", "description": "Replace exact text in a file once.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {
     "name": "glob", "description": "Find files matching a glob pattern.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
]

TOOL_HANDLERS = {
    "bash": run_bash,
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "glob": run_glob,
}

# ── The core pattern: a while loop that calls tools until the model stops ──
def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        choice = response.choices[0]
        assistant_msg = choice.message
        # Append assistant turn
        msg = {"role": "assistant", "content": assistant_msg.content}
        if assistant_msg.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tool.id,
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "arguments": tool.function.arguments,
                    },
                }
                for tool in assistant_msg.tool_calls
            ]
        messages.append(msg)

        # If the model didn't call a tool, we're done
        if choice.finish_reason != "tool_calls":
            return

        # Execute each tool call, collect results
        for tool in assistant_msg.tool_calls:
            args = json.loads(tool.function.arguments)
            handler = TOOL_HANDLERS.get(tool.function.name)
            if handler:
                output = handler(**args)
            else:
                output = f"Error: no handler for {tool.function.name}"
            print(f"\033[33m$ {tool.function.name} {args}\033[0m")
            print(output[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "content": output,
            })


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    print("test01: Agent Loop")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36mBAgent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", "bye","退出","再见"):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response
        respones_content = history[-1]
        if respones_content["role"] == "assistant" and respones_content.get("content"):
            print(respones_content["content"])
        print()
