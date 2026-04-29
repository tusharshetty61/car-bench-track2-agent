"""Synchronous client for A2A communication - safe to use in thread pools."""
import json
from uuid import uuid4

import httpx
from a2a.types import Message, Part, Role, TextPart, DataPart


DEFAULT_TIMEOUT = 300


def create_message_with_parts(*, role: Role = Role.user, parts: list[Part], context_id: str | None = None, task_id: str | None = None) -> Message:
    """Create a message with custom parts."""
    return Message(
        kind="message",
        role=role,
        parts=parts,
        message_id=uuid4().hex,
        context_id=context_id,
        task_id=task_id,
    )


def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


def send_message_with_parts_sync(parts: list[Part], base_url: str, context_id: str | None = None, task_id: str | None = None) -> dict:
    """Send a message with custom parts synchronously. Safe for use in thread pools.
    
    Returns dict with context_id, task_id, response and status (if exists)
    """
    # Create the message
    outbound_msg = create_message_with_parts(parts=parts, context_id=context_id, task_id=task_id)
    
    # Prepare JSON-RPC request
    jsonrpc_request = {
        "jsonrpc": "2.0",
        "id": uuid4().hex,
        "method": "message/send",
        "params": {
            "message": outbound_msg.model_dump(mode="json", exclude_none=True)
        }
    }
    
    # Use synchronous httpx client
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        response = client.post(
            base_url,
            json=jsonrpc_request,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "error" in result:
            raise RuntimeError(f"JSON-RPC error: {result['error']}")
        
        response_data = result.get("result", {})
        
        # Parse the response based on type
        outputs = {
            "response": "",
            "context_id": None,
            "task_id": None,
            "raw_message": None
        }
        
        # Check if it's a Message or Task response
        if response_data.get("kind") == "message":
            # Direct message response
            msg = Message.model_validate(response_data)
            outputs["context_id"] = msg.context_id
            outputs["task_id"] = msg.task_id
            outputs["response"] = merge_parts(msg.parts)
            outputs["raw_message"] = msg
        elif response_data.get("kind") == "task":
            # Task response
            from a2a.types import Task
            task = Task.model_validate(response_data)
            outputs["context_id"] = task.context_id
            outputs["task_id"] = task.id
            outputs["status"] = task.status.state.value
            if task.status.message:
                outputs["response"] = merge_parts(task.status.message.parts)
                outputs["raw_message"] = task.status.message
            if task.artifacts:
                for artifact in task.artifacts:
                    outputs["response"] += merge_parts(artifact.parts)
        
        return outputs
