from agents import Runner, RunConfig, TResponseInputItem, trace
from pydantic import BaseModel
from agent import my_agent

class WorkflowInput(BaseModel):
    input_as_text: str

async def run_workflow(workflow_input: WorkflowInput):
    with trace("Equinix Agent"):
        conversation_history = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": workflow_input.input_as_text}]
            }
        ]

        result = await Runner.run(
            my_agent,
            input=[*conversation_history],
            run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69135893f40c819095704afbaed0bf0e0d3e74f0b6d2392c"
            })
        )

        return {"output_text": result.final_output_as(str)}
