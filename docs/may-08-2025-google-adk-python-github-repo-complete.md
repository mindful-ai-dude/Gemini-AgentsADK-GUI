Directory structure:
└── google-adk-python-github-repo-complete/
    ├── __init__.py
    ├── runners.py
    ├── telemetry.py
    ├── version.py
    ├── agents/
    │   ├── __init__.py
    │   ├── active_streaming_tool.py
    │   ├── base_agent.py
    │   ├── callback_context.py
    │   ├── invocation_context.py
    │   ├── langgraph_agent.py
    │   ├── live_request_queue.py
    │   ├── llm_agent.py
    │   ├── loop_agent.py
    │   ├── parallel_agent.py
    │   ├── readonly_context.py
    │   ├── remote_agent.py
    │   ├── run_config.py
    │   ├── sequential_agent.py
    │   └── transcription_entry.py
    ├── artifacts/
    │   ├── __init__.py
    │   ├── base_artifact_service.py
    │   ├── gcs_artifact_service.py
    │   └── in_memory_artifact_service.py
    ├── auth/
    │   ├── __init__.py
    │   ├── auth_credential.py
    │   ├── auth_handler.py
    │   ├── auth_preprocessor.py
    │   ├── auth_schemes.py
    │   └── auth_tool.py
    ├── cli/
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── agent_graph.py
    │   ├── cli.py
    │   ├── cli_create.py
    │   ├── cli_deploy.py
    │   ├── cli_eval.py
    │   ├── cli_tools_click.py
    │   ├── fast_api.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── envs.py
    │       ├── evals.py
    │       └── logs.py
    ├── code_executors/
    │   ├── __init__.py
    │   ├── base_code_executor.py
    │   ├── code_execution_utils.py
    │   ├── code_executor_context.py
    │   ├── container_code_executor.py
    │   ├── unsafe_local_code_executor.py
    │   └── vertex_ai_code_executor.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── agent_evaluator.py
    │   ├── evaluation_constants.py
    │   ├── evaluation_generator.py
    │   ├── response_evaluator.py
    │   └── trajectory_evaluator.py
    ├── events/
    │   ├── __init__.py
    │   ├── event.py
    │   └── event_actions.py
    ├── examples/
    │   ├── __init__.py
    │   ├── base_example_provider.py
    │   ├── example.py
    │   ├── example_util.py
    │   └── vertex_ai_example_store.py
    ├── flows/
    │   ├── __init__.py
    │   └── llm_flows/
    │       ├── __init__.py
    │       ├── _base_llm_processor.py
    │       ├── _code_execution.py
    │       ├── _nl_planning.py
    │       ├── agent_transfer.py
    │       ├── audio_transcriber.py
    │       ├── auto_flow.py
    │       ├── base_llm_flow.py
    │       ├── basic.py
    │       ├── contents.py
    │       ├── functions.py
    │       ├── identity.py
    │       ├── instructions.py
    │       └── single_flow.py
    ├── memory/
    │   ├── __init__.py
    │   ├── base_memory_service.py
    │   ├── in_memory_memory_service.py
    │   └── vertex_ai_rag_memory_service.py
    ├── models/
    │   ├── __init__.py
    │   ├── anthropic_llm.py
    │   ├── base_llm.py
    │   ├── base_llm_connection.py
    │   ├── gemini_llm_connection.py
    │   ├── google_llm.py
    │   ├── lite_llm.py
    │   ├── llm_request.py
    │   ├── llm_response.py
    │   └── registry.py
    ├── planners/
    │   ├── __init__.py
    │   ├── base_planner.py
    │   ├── built_in_planner.py
    │   └── plan_re_act_planner.py
    ├── sessions/
    │   ├── __init__.py
    │   ├── _session_util.py
    │   ├── base_session_service.py
    │   ├── database_session_service.py
    │   ├── in_memory_session_service.py
    │   ├── session.py
    │   ├── state.py
    │   └── vertex_ai_session_service.py
    └── tools/
        ├── __init__.py
        ├── _automatic_function_calling_util.py
        ├── agent_tool.py
        ├── base_tool.py
        ├── built_in_code_execution_tool.py
        ├── crewai_tool.py
        ├── example_tool.py
        ├── exit_loop_tool.py
        ├── function_parameter_parse_util.py
        ├── function_tool.py
        ├── get_user_choice_tool.py
        ├── google_search_tool.py
        ├── langchain_tool.py
        ├── load_artifacts_tool.py
        ├── load_memory_tool.py
        ├── load_web_page.py
        ├── long_running_tool.py
        ├── preload_memory_tool.py
        ├── tool_context.py
        ├── toolbox_tool.py
        ├── transfer_to_agent_tool.py
        ├── vertex_ai_search_tool.py
        ├── apihub_tool/
        │   ├── __init__.py
        │   ├── apihub_toolset.py
        │   └── clients/
        │       ├── __init__.py
        │       ├── apihub_client.py
        │       └── secret_client.py
        ├── application_integration_tool/
        │   ├── __init__.py
        │   ├── application_integration_toolset.py
        │   ├── integration_connector_tool.py
        │   └── clients/
        │       ├── connections_client.py
        │       └── integration_client.py
        ├── google_api_tool/
        │   ├── __init__.py
        │   ├── google_api_tool.py
        │   ├── google_api_tool_set.py
        │   ├── google_api_tool_sets.py
        │   └── googleapi_to_openapi_converter.py
        ├── mcp_tool/
        │   ├── __init__.py
        │   ├── conversion_utils.py
        │   ├── mcp_session_manager.py
        │   ├── mcp_tool.py
        │   └── mcp_toolset.py
        ├── openapi_tool/
        │   ├── __init__.py
        │   ├── auth/
        │   │   ├── __init__.py
        │   │   ├── auth_helpers.py
        │   │   └── credential_exchangers/
        │   │       ├── __init__.py
        │   │       ├── auto_auth_credential_exchanger.py
        │   │       ├── base_credential_exchanger.py
        │   │       ├── oauth2_exchanger.py
        │   │       └── service_account_exchanger.py
        │   ├── common/
        │   │   ├── __init__.py
        │   │   └── common.py
        │   └── openapi_spec_parser/
        │       ├── __init__.py
        │       ├── openapi_spec_parser.py
        │       ├── openapi_toolset.py
        │       ├── operation_parser.py
        │       ├── rest_api_tool.py
        │       └── tool_auth_handler.py
        └── retrieval/
            ├── __init__.py
            ├── base_retrieval_tool.py
            ├── files_retrieval.py
            ├── llama_index_retrieval.py
            └── vertex_ai_rag_retrieval.py

================================================
FILE: src/google/adk/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import version
from .agents.llm_agent import Agent
from .runners import Runner

__version__ = version.__version__
__all__ = ["Agent", "Runner"]



================================================
FILE: src/google/adk/runners.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import AsyncGenerator
from typing import Generator
from typing import Optional

from deprecated import deprecated
from google.genai import types

from .agents.active_streaming_tool import ActiveStreamingTool
from .agents.base_agent import BaseAgent
from .agents.invocation_context import InvocationContext
from .agents.invocation_context import new_invocation_context_id
from .agents.live_request_queue import LiveRequestQueue
from .agents.llm_agent import LlmAgent
from .agents.run_config import RunConfig
from .agents.run_config import StreamingMode
from .artifacts.base_artifact_service import BaseArtifactService
from .artifacts.in_memory_artifact_service import InMemoryArtifactService
from .events.event import Event
from .memory.base_memory_service import BaseMemoryService
from .memory.in_memory_memory_service import InMemoryMemoryService
from .sessions.base_session_service import BaseSessionService
from .sessions.in_memory_session_service import InMemorySessionService
from .sessions.session import Session
from .telemetry import tracer
from .tools.built_in_code_execution_tool import built_in_code_execution

logger = logging.getLogger(__name__)


class Runner:
  """The Runner class is used to run agents.

  It manages the execution of an agent within a session, handling message
  processing, event generation, and interaction with various services like
  artifact storage, session management, and memory.

  Attributes:
      app_name: The application name of the runner.
      agent: The root agent to run.
      artifact_service: The artifact service for the runner.
      session_service: The session service for the runner.
      memory_service: The memory service for the runner.
  """

  app_name: str
  """The app name of the runner."""
  agent: BaseAgent
  """The root agent to run."""
  artifact_service: Optional[BaseArtifactService] = None
  """The artifact service for the runner."""
  session_service: BaseSessionService
  """The session service for the runner."""
  memory_service: Optional[BaseMemoryService] = None
  """The memory service for the runner."""

  def __init__(
      self,
      *,
      app_name: str,
      agent: BaseAgent,
      artifact_service: Optional[BaseArtifactService] = None,
      session_service: BaseSessionService,
      memory_service: Optional[BaseMemoryService] = None,
  ):
    """Initializes the Runner.

    Args:
        app_name: The application name of the runner.
        agent: The root agent to run.
        artifact_service: The artifact service for the runner.
        session_service: The session service for the runner.
        memory_service: The memory service for the runner.
    """
    self.app_name = app_name
    self.agent = agent
    self.artifact_service = artifact_service
    self.session_service = session_service
    self.memory_service = memory_service

  def run(
      self,
      *,
      user_id: str,
      session_id: str,
      new_message: types.Content,
      run_config: RunConfig = RunConfig(),
  ) -> Generator[Event, None, None]:
    """Runs the agent.

    NOTE: This sync interface is only for local testing and convenience purpose.
    Consider using `run_async` for production usage.

    Args:
      user_id: The user ID of the session.
      session_id: The session ID of the session.
      new_message: A new message to append to the session.
      run_config: The run config for the agent.

    Yields:
      The events generated by the agent.
    """
    event_queue = queue.Queue()

    async def _invoke_run_async():
      try:
        async for event in self.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
            run_config=run_config,
        ):
          event_queue.put(event)
      finally:
        event_queue.put(None)

    def _asyncio_thread_main():
      try:
        asyncio.run(_invoke_run_async())
      finally:
        event_queue.put(None)

    thread = threading.Thread(target=_asyncio_thread_main)
    thread.start()

    # consumes and re-yield the events from background thread.
    while True:
      event = event_queue.get()
      if event is None:
        break
      else:
        yield event

    thread.join()

  async def run_async(
      self,
      *,
      user_id: str,
      session_id: str,
      new_message: types.Content,
      run_config: RunConfig = RunConfig(),
  ) -> AsyncGenerator[Event, None]:
    """Main entry method to run the agent in this runner.

    Args:
      user_id: The user ID of the session.
      session_id: The session ID of the session.
      new_message: A new message to append to the session.
      run_config: The run config for the agent.

    Yields:
      The events generated by the agent.
    """
    with tracer.start_as_current_span('invocation'):
      session = self.session_service.get_session(
          app_name=self.app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        raise ValueError(f'Session not found: {session_id}')

      invocation_context = self._new_invocation_context(
          session,
          new_message=new_message,
          run_config=run_config,
      )
      root_agent = self.agent

      if new_message:
        await self._append_new_message_to_session(
            session,
            new_message,
            invocation_context,
            run_config.save_input_blobs_as_artifacts,
        )

      invocation_context.agent = self._find_agent_to_run(session, root_agent)
      async for event in invocation_context.agent.run_async(invocation_context):
        if not event.partial:
          self.session_service.append_event(session=session, event=event)
        yield event

  async def _append_new_message_to_session(
      self,
      session: Session,
      new_message: types.Content,
      invocation_context: InvocationContext,
      save_input_blobs_as_artifacts: bool = False,
  ):
    """Appends a new message to the session.

    Args:
        session: The session to append the message to.
        new_message: The new message to append.
        invocation_context: The invocation context for the message.
        save_input_blobs_as_artifacts: Whether to save input blobs as artifacts.
    """
    if not new_message.parts:
      raise ValueError('No parts in the new_message.')

    if self.artifact_service and save_input_blobs_as_artifacts:
      # The runner directly saves the artifacts (if applicable) in the
      # user message and replaces the artifact data with a file name
      # placeholder.
      for i, part in enumerate(new_message.parts):
        if part.inline_data is None:
          continue
        file_name = f'artifact_{invocation_context.invocation_id}_{i}'
        await self.artifact_service.save_artifact(
            app_name=self.app_name,
            user_id=session.user_id,
            session_id=session.id,
            filename=file_name,
            artifact=part,
        )
        new_message.parts[i] = types.Part(
            text=f'Uploaded file: {file_name}. It is saved into artifacts'
        )
    # Appends only. We do not yield the event because it's not from the model.
    event = Event(
        invocation_id=invocation_context.invocation_id,
        author='user',
        content=new_message,
    )
    self.session_service.append_event(session=session, event=event)

  async def run_live(
      self,
      *,
      session: Session,
      live_request_queue: LiveRequestQueue,
      run_config: RunConfig = RunConfig(),
  ) -> AsyncGenerator[Event, None]:
    """Runs the agent in live mode (experimental feature).

    Args:
        session: The session to use.
        live_request_queue: The queue for live requests.
        run_config: The run config for the agent.

    Yields:
        The events generated by the agent.

    .. warning::
        This feature is **experimental** and its API or behavior may change
        in future releases.
    """
    # TODO: right now, only works for a single audio agent without FC.
    invocation_context = self._new_invocation_context_for_live(
        session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )

    root_agent = self.agent
    invocation_context.agent = self._find_agent_to_run(session, root_agent)

    invocation_context.active_streaming_tools = {}
    # TODO(hangfei): switch to use canonical_tools.
    for tool in invocation_context.agent.tools:
      # replicate a LiveRequestQueue for streaming tools that relis on
      # LiveRequestQueue
      from typing import get_type_hints

      type_hints = get_type_hints(tool)
      for arg_type in type_hints.values():
        if arg_type is LiveRequestQueue:
          if not invocation_context.active_streaming_tools:
            invocation_context.active_streaming_tools = {}
          active_streaming_tools = ActiveStreamingTool(
              stream=LiveRequestQueue()
          )
          invocation_context.active_streaming_tools[tool.__name__] = (
              active_streaming_tools
          )

    async for event in invocation_context.agent.run_live(invocation_context):
      self.session_service.append_event(session=session, event=event)
      yield event

  async def close_session(self, session: Session):
    """Closes a session and adds it to the memory service (experimental feature).

    Args:
        session: The session to close.
    """
    if self.memory_service:
      await self.memory_service.add_session_to_memory(session)
    self.session_service.close_session(session=session)

  def _find_agent_to_run(
      self, session: Session, root_agent: BaseAgent
  ) -> BaseAgent:
    """Finds the agent to run to continue the session.

    A qualified agent must be either of:
    - The root agent;
    - An LlmAgent who replied last and is capable to transfer to any other agent
      in the agent hierarchy.

    Args:
        session: The session to find the agent for.
        root_agent: The root agent of the runner.

    Returns:
      The agent of the last message in the session or the root agent.
    """
    for event in filter(lambda e: e.author != 'user', reversed(session.events)):
      if event.author == root_agent.name:
        # Found root agent.
        return root_agent
      if not (agent := root_agent.find_sub_agent(event.author)):
        # Agent not found, continue looking.
        logger.warning(
            'Event from an unknown agent: %s, event id: %s',
            event.author,
            event.id,
        )
        continue
      if self._is_transferable_across_agent_tree(agent):
        return agent
    # Falls back to root agent if no suitable agents are found in the session.
    return root_agent

  def _is_transferable_across_agent_tree(self, agent_to_run: BaseAgent) -> bool:
    """Whether the agent to run can transfer to any other agent in the agent tree.

    This typically means all agent_to_run's parent through root agent can
    transfer to their parent_agent.

    Args:
        agent_to_run: The agent to check for transferability.

    Returns:
        True if the agent can transfer, False otherwise.
    """
    agent = agent_to_run
    while agent:
      if not isinstance(agent, LlmAgent):
        # Only LLM-based Agent can provider agent transfer capability.
        return False
      if agent.disallow_transfer_to_parent:
        return False
      agent = agent.parent_agent
    return True

  def _new_invocation_context(
      self,
      session: Session,
      *,
      new_message: Optional[types.Content] = None,
      live_request_queue: Optional[LiveRequestQueue] = None,
      run_config: RunConfig = RunConfig(),
  ) -> InvocationContext:
    """Creates a new invocation context.

    Args:
        session: The session for the context.
        new_message: The new message for the context.
        live_request_queue: The live request queue for the context.
        run_config: The run config for the context.

    Returns:
        The new invocation context.
    """
    invocation_id = new_invocation_context_id()

    if run_config.support_cfc and isinstance(self.agent, LlmAgent):
      model_name = self.agent.canonical_model.model
      if not model_name.startswith('gemini-2'):
        raise ValueError(
            f'CFC is not supported for model: {model_name} in agent:'
            f' {self.agent.name}'
        )
      if built_in_code_execution not in self.agent.canonical_tools:
        self.agent.tools.append(built_in_code_execution)

    return InvocationContext(
        artifact_service=self.artifact_service,
        session_service=self.session_service,
        memory_service=self.memory_service,
        invocation_id=invocation_id,
        agent=self.agent,
        session=session,
        user_content=new_message,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )

  def _new_invocation_context_for_live(
      self,
      session: Session,
      *,
      live_request_queue: Optional[LiveRequestQueue] = None,
      run_config: RunConfig = RunConfig(),
  ) -> InvocationContext:
    """Creates a new invocation context for live multi-agent."""

    # For live multi-agent, we need model's text transcription as context for
    # next agent.
    if self.agent.sub_agents and live_request_queue:
      if not run_config.response_modalities:
        # default
        run_config.response_modalities = ['AUDIO']
        if not run_config.output_audio_transcription:
          run_config.output_audio_transcription = (
              types.AudioTranscriptionConfig()
          )
      elif 'TEXT' not in run_config.response_modalities:
        if not run_config.output_audio_transcription:
          run_config.output_audio_transcription = (
              types.AudioTranscriptionConfig()
          )
    return self._new_invocation_context(
        session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )


class InMemoryRunner(Runner):
  """An in-memory Runner for testing and development.

  This runner uses in-memory implementations for artifact, session, and memory
  services, providing a lightweight and self-contained environment for agent
  execution.

  Attributes:
      agent: The root agent to run.
      app_name: The application name of the runner. Defaults to
        'InMemoryRunner'.
  """

  def __init__(self, agent: LlmAgent, *, app_name: str = 'InMemoryRunner'):
    """Initializes the InMemoryRunner.

    Args:
        agent: The root agent to run.
        app_name: The application name of the runner. Defaults to
          'InMemoryRunner'.
    """
    super().__init__(
        app_name=app_name,
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )



================================================
FILE: src/google/adk/telemetry.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE:
#
#    We expect that the underlying GenAI SDK will provide a certain
#    level of tracing and logging telemetry aligned with Open Telemetry
#    Semantic Conventions (such as logging prompts, responses,
#    request properties, etc.) and so the information that is recorded by the
#    Agent Development Kit should be focused on the higher-level
#    constructs of the framework that are not observable by the SDK.

import json
from typing import Any

from google.genai import types
from opentelemetry import trace

from .agents.invocation_context import InvocationContext
from .events.event import Event
from .models.llm_request import LlmRequest
from .models.llm_response import LlmResponse


tracer = trace.get_tracer('gcp.vertex.agent')


def trace_tool_call(
    args: dict[str, Any],
):
  """Traces tool call.

  Args:
    args: The arguments to the tool call.
  """
  span = trace.get_current_span()
  span.set_attribute('gen_ai.system', 'gcp.vertex.agent')
  span.set_attribute('gcp.vertex.agent.tool_call_args', json.dumps(args))


def trace_tool_response(
    invocation_context: InvocationContext,
    event_id: str,
    function_response_event: Event,
):
  """Traces tool response event.

  This function records details about the tool response event as attributes on
  the current OpenTelemetry span.

  Args:
    invocation_context: The invocation context for the current agent run.
    event_id: The ID of the event.
    function_response_event: The function response event which can be either
      merged function response for parallel function calls or individual
      function response for sequential function calls.
  """
  span = trace.get_current_span()
  span.set_attribute('gen_ai.system', 'gcp.vertex.agent')
  span.set_attribute(
      'gcp.vertex.agent.invocation_id', invocation_context.invocation_id
  )
  span.set_attribute('gcp.vertex.agent.event_id', event_id)
  span.set_attribute(
      'gcp.vertex.agent.tool_response',
      function_response_event.model_dump_json(exclude_none=True),
  )

  # Setting empty llm request and response (as UI expect these) while not
  # applicable for tool_response.
  span.set_attribute('gcp.vertex.agent.llm_request', '{}')
  span.set_attribute(
      'gcp.vertex.agent.llm_response',
      '{}',
  )


def trace_call_llm(
    invocation_context: InvocationContext,
    event_id: str,
    llm_request: LlmRequest,
    llm_response: LlmResponse,
):
  """Traces a call to the LLM.

  This function records details about the LLM request and response as
  attributes on the current OpenTelemetry span.

  Args:
    invocation_context: The invocation context for the current agent run.
    event_id: The ID of the event.
    llm_request: The LLM request object.
    llm_response: The LLM response object.
  """
  span = trace.get_current_span()
  # Special standard Open Telemetry GenaI attributes that indicate
  # that this is a span related to a Generative AI system.
  span.set_attribute('gen_ai.system', 'gcp.vertex.agent')
  span.set_attribute('gen_ai.request.model', llm_request.model)
  span.set_attribute(
      'gcp.vertex.agent.invocation_id', invocation_context.invocation_id
  )
  span.set_attribute('gcp.vertex.agent.event_id', event_id)
  # Consider removing once GenAI SDK provides a way to record this info.
  span.set_attribute(
      'gcp.vertex.agent.llm_request',
      json.dumps(_build_llm_request_for_trace(llm_request)),
  )
  # Consider removing once GenAI SDK provides a way to record this info.
  span.set_attribute(
      'gcp.vertex.agent.llm_response',
      llm_response.model_dump_json(exclude_none=True),
  )


def trace_send_data(
    invocation_context: InvocationContext,
    event_id: str,
    data: list[types.Content],
):
  """Traces the sending of data to the agent.

  This function records details about the data sent to the agent as
  attributes on the current OpenTelemetry span.

  Args:
    invocation_context: The invocation context for the current agent run.
    event_id: The ID of the event.
    data: A list of content objects.
  """
  span = trace.get_current_span()
  span.set_attribute(
      'gcp.vertex.agent.invocation_id', invocation_context.invocation_id
  )
  span.set_attribute('gcp.vertex.agent.event_id', event_id)
  # Once instrumentation is added to the GenAI SDK, consider whether this
  # information still needs to be recorded by the Agent Development Kit.
  span.set_attribute(
      'gcp.vertex.agent.data',
      json.dumps([
          types.Content(role=content.role, parts=content.parts).model_dump(
              exclude_none=True
          )
          for content in data
      ]),
  )


def _build_llm_request_for_trace(llm_request: LlmRequest) -> dict[str, Any]:
  """Builds a dictionary representation of the LLM request for tracing.

  This function prepares a dictionary representation of the LlmRequest
  object, suitable for inclusion in a trace. It excludes fields that cannot
  be serialized (e.g., function pointers) and avoids sending bytes data.

  Args:
    llm_request: The LlmRequest object.

  Returns:
    A dictionary representation of the LLM request.
  """
  # Some fields in LlmRequest are function pointers and can not be serialized.
  result = {
      'model': llm_request.model,
      'config': llm_request.config.model_dump(
          exclude_none=True, exclude='response_schema'
      ),
      'contents': [],
  }
  # We do not want to send bytes data to the trace.
  for content in llm_request.contents:
    parts = [part for part in content.parts if not part.inline_data]
    result['contents'].append(
        types.Content(role=content.role, parts=parts).model_dump(
            exclude_none=True
        )
    )
  return result



================================================
FILE: src/google/adk/version.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# version: date+base_cl
__version__ = "0.4.0"



================================================
FILE: src/google/adk/agents/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base_agent import BaseAgent
from .live_request_queue import LiveRequest
from .live_request_queue import LiveRequestQueue
from .llm_agent import Agent
from .llm_agent import LlmAgent
from .loop_agent import LoopAgent
from .parallel_agent import ParallelAgent
from .run_config import RunConfig
from .sequential_agent import SequentialAgent

__all__ = [
    'Agent',
    'BaseAgent',
    'LlmAgent',
    'LoopAgent',
    'ParallelAgent',
    'SequentialAgent',
]



================================================
FILE: src/google/adk/agents/active_streaming_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from .live_request_queue import LiveRequestQueue


class ActiveStreamingTool(BaseModel):
  """Manages streaming tool related resources during invocation."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  task: Optional[asyncio.Task] = None
  """The active task of this streaming tool."""

  stream: Optional[LiveRequestQueue] = None
  """The active (input) streams of this streaming tool."""



================================================
FILE: src/google/adk/agents/base_agent.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Union
from typing import AsyncGenerator
from typing import Callable
from typing import final
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from opentelemetry import trace
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from typing_extensions import override

from ..events.event import Event
from .callback_context import CallbackContext

if TYPE_CHECKING:
  from .invocation_context import InvocationContext

tracer = trace.get_tracer('gcp.vertex.agent')

BeforeAgentCallback = Callable[
    [CallbackContext],
    Union[Awaitable[Optional[types.Content]], Optional[types.Content]],
]

AfterAgentCallback = Callable[
    [CallbackContext],
    Union[Awaitable[Optional[types.Content]], Optional[types.Content]],
]


class BaseAgent(BaseModel):
  """Base class for all agents in Agent Development Kit."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  name: str
  """The agent's name.

  Agent name must be a Python identifier and unique within the agent tree.
  Agent name cannot be "user", since it's reserved for end-user's input.
  """

  description: str = ''
  """Description about the agent's capability.

  The model uses this to determine whether to delegate control to the agent.
  One-line description is enough and preferred.
  """

  parent_agent: Optional[BaseAgent] = Field(default=None, init=False)
  """The parent agent of this agent.

  Note that an agent can ONLY be added as sub-agent once.

  If you want to add one agent twice as sub-agent, consider to create two agent
  instances with identical config, but with different name and add them to the
  agent tree.
  """
  sub_agents: list[BaseAgent] = Field(default_factory=list)
  """The sub-agents of this agent."""

  before_agent_callback: Optional[BeforeAgentCallback] = None
  """Callback signature that is invoked before the agent run.

  Args:
    callback_context: MUST be named 'callback_context' (enforced).

  Returns:
    Optional[types.Content]: The content to return to the user.
      When the content is present, the agent run will be skipped and the
      provided content will be returned to user.
  """
  after_agent_callback: Optional[AfterAgentCallback] = None
  """Callback signature that is invoked after the agent run.

  Args:
    callback_context: MUST be named 'callback_context' (enforced).

  Returns:
    Optional[types.Content]: The content to return to the user.
      When the content is present, the provided content will be used as agent
      response and appended to event history as agent response.
  """

  @final
  async def run_async(
      self,
      parent_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Entry method to run an agent via text-based conversation.

    Args:
      parent_context: InvocationContext, the invocation context of the parent
        agent.

    Yields:
      Event: the events generated by the agent.
    """

    with tracer.start_as_current_span(f'agent_run [{self.name}]'):
      ctx = self._create_invocation_context(parent_context)

      if event := await self.__handle_before_agent_callback(ctx):
        yield event
      if ctx.end_invocation:
        return

      async for event in self._run_async_impl(ctx):
        yield event

      if ctx.end_invocation:
        return

      if event := await self.__handle_after_agent_callback(ctx):
        yield event

  @final
  async def run_live(
      self,
      parent_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Entry method to run an agent via video/audio-based conversation.

    Args:
      parent_context: InvocationContext, the invocation context of the parent
        agent.

    Yields:
      Event: the events generated by the agent.
    """
    with tracer.start_as_current_span(f'agent_run [{self.name}]'):
      ctx = self._create_invocation_context(parent_context)
      # TODO(hangfei): support before/after_agent_callback

      async for event in self._run_live_impl(ctx):
        yield event

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Core logic to run this agent via text-based conversation.

    Args:
      ctx: InvocationContext, the invocation context for this agent.

    Yields:
      Event: the events generated by the agent.
    """
    raise NotImplementedError(
        f'_run_async_impl for {type(self)} is not implemented.'
    )
    yield  # AsyncGenerator requires having at least one yield statement

  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Core logic to run this agent via video/audio-based conversation.

    Args:
      ctx: InvocationContext, the invocation context for this agent.

    Yields:
      Event: the events generated by the agent.
    """
    raise NotImplementedError(
        f'_run_live_impl for {type(self)} is not implemented.'
    )
    yield  # AsyncGenerator requires having at least one yield statement

  @property
  def root_agent(self) -> BaseAgent:
    """Gets the root agent of this agent."""
    root_agent = self
    while root_agent.parent_agent is not None:
      root_agent = root_agent.parent_agent
    return root_agent

  def find_agent(self, name: str) -> Optional[BaseAgent]:
    """Finds the agent with the given name in this agent and its descendants.

    Args:
      name: The name of the agent to find.

    Returns:
      The agent with the matching name, or None if no such agent is found.
    """
    if self.name == name:
      return self
    return self.find_sub_agent(name)

  def find_sub_agent(self, name: str) -> Optional[BaseAgent]:
    """Finds the agent with the given name in this agent's descendants.

    Args:
      name: The name of the agent to find.

    Returns:
      The agent with the matching name, or None if no such agent is found.
    """
    for sub_agent in self.sub_agents:
      if result := sub_agent.find_agent(name):
        return result
    return None

  def _create_invocation_context(
      self, parent_context: InvocationContext
  ) -> InvocationContext:
    """Creates a new invocation context for this agent."""
    invocation_context = parent_context.model_copy(update={'agent': self})
    if parent_context.branch:
      invocation_context.branch = f'{parent_context.branch}.{self.name}'
    return invocation_context

  async def __handle_before_agent_callback(
      self, ctx: InvocationContext
  ) -> Optional[Event]:
    """Runs the before_agent_callback if it exists.

    Returns:
      Optional[Event]: an event if callback provides content or changed state.
    """
    ret_event = None

    if not isinstance(self.before_agent_callback, Callable):
      return ret_event

    callback_context = CallbackContext(ctx)
    before_agent_callback_content = self.before_agent_callback(
        callback_context=callback_context
    )

    if inspect.isawaitable(before_agent_callback_content):
      before_agent_callback_content = await before_agent_callback_content

    if before_agent_callback_content:
      ret_event = Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          branch=ctx.branch,
          content=before_agent_callback_content,
          actions=callback_context._event_actions,
      )
      ctx.end_invocation = True
      return ret_event

    if callback_context.state.has_delta():
      ret_event = Event(
          invocation_id=ctx.invocation_id,
          author=self.name,
          branch=ctx.branch,
          actions=callback_context._event_actions,
      )

    return ret_event

  async def __handle_after_agent_callback(
      self, invocation_context: InvocationContext
  ) -> Optional[Event]:
    """Runs the after_agent_callback if it exists.

    Returns:
      Optional[Event]: an event if callback provides content or changed state.
    """
    ret_event = None

    if not isinstance(self.after_agent_callback, Callable):
      return ret_event

    callback_context = CallbackContext(invocation_context)
    after_agent_callback_content = self.after_agent_callback(
        callback_context=callback_context
    )

    if inspect.isawaitable(after_agent_callback_content):
      after_agent_callback_content = await after_agent_callback_content

    if after_agent_callback_content or callback_context.state.has_delta():
      ret_event = Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          branch=invocation_context.branch,
          content=after_agent_callback_content,
          actions=callback_context._event_actions,
      )

    return ret_event

  @override
  def model_post_init(self, __context: Any) -> None:
    self.__set_parent_agent_for_sub_agents()

  @field_validator('name', mode='after')
  @classmethod
  def __validate_name(cls, value: str):
    if not value.isidentifier():
      raise ValueError(
          f'Found invalid agent name: `{value}`.'
          ' Agent name must be a valid identifier. It should start with a'
          ' letter (a-z, A-Z) or an underscore (_), and can only contain'
          ' letters, digits (0-9), and underscores.'
      )
    if value == 'user':
      raise ValueError(
          "Agent name cannot be `user`. `user` is reserved for end-user's"
          ' input.'
      )
    return value

  def __set_parent_agent_for_sub_agents(self) -> BaseAgent:
    for sub_agent in self.sub_agents:
      if sub_agent.parent_agent is not None:
        raise ValueError(
            f'Agent `{sub_agent.name}` already has a parent agent, current'
            f' parent: `{sub_agent.parent_agent.name}`, trying to add:'
            f' `{self.name}`'
        )
      sub_agent.parent_agent = self
    return self



================================================
FILE: src/google/adk/agents/callback_context.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from typing_extensions import override

from .readonly_context import ReadonlyContext

if TYPE_CHECKING:
  from google.genai import types

  from ..events.event_actions import EventActions
  from ..sessions.state import State
  from .invocation_context import InvocationContext


class CallbackContext(ReadonlyContext):
  """The context of various callbacks within an agent run."""

  def __init__(
      self,
      invocation_context: InvocationContext,
      *,
      event_actions: Optional[EventActions] = None,
  ) -> None:
    super().__init__(invocation_context)

    from ..events.event_actions import EventActions
    from ..sessions.state import State

    # TODO(weisun): make this public for Agent Development Kit, but private for
    # users.
    self._event_actions = event_actions or EventActions()
    self._state = State(
        value=invocation_context.session.state,
        delta=self._event_actions.state_delta,
    )

  @property
  @override
  def state(self) -> State:
    """The delta-aware state of the current session.

    For any state change, you can mutate this object directly,
    e.g. `ctx.state['foo'] = 'bar'`
    """
    return self._state

  @property
  def user_content(self) -> Optional[types.Content]:
    """The user content that started this invocation. READONLY field."""
    return self._invocation_context.user_content

  async def load_artifact(
      self, filename: str, version: Optional[int] = None
  ) -> Optional[types.Part]:
    """Loads an artifact attached to the current session.

    Args:
      filename: The filename of the artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError("Artifact service is not initialized.")
    return await self._invocation_context.artifact_service.load_artifact(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        version=version,
    )

  async def save_artifact(self, filename: str, artifact: types.Part) -> int:
    """Saves an artifact and records it as delta for the current session.

    Args:
      filename: The filename of the artifact.
      artifact: The artifact to save.

    Returns:
     The version of the artifact.
    """
    if self._invocation_context.artifact_service is None:
      raise ValueError("Artifact service is not initialized.")
    version = await self._invocation_context.artifact_service.save_artifact(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
        filename=filename,
        artifact=artifact,
    )
    self._event_actions.artifact_delta[filename] = version
    return version



================================================
FILE: src/google/adk/agents/invocation_context.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional
import uuid

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict

from ..artifacts.base_artifact_service import BaseArtifactService
from ..memory.base_memory_service import BaseMemoryService
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from .active_streaming_tool import ActiveStreamingTool
from .base_agent import BaseAgent
from .live_request_queue import LiveRequestQueue
from .run_config import RunConfig
from .transcription_entry import TranscriptionEntry


class LlmCallsLimitExceededError(Exception):
  """Error thrown when the number of LLM calls exceed the limit."""


class _InvocationCostManager(BaseModel):
  """A container to keep track of the cost of invocation.

  While we don't expected the metrics captured here to be a direct
  representatative of monetary cost incurred in executing the current
  invocation, but they, in someways have an indirect affect.
  """

  _number_of_llm_calls: int = 0
  """A counter that keeps track of number of llm calls made."""

  def increment_and_enforce_llm_calls_limit(
      self, run_config: Optional[RunConfig]
  ):
    """Increments _number_of_llm_calls and enforces the limit."""
    # We first increment the counter and then check the conditions.
    self._number_of_llm_calls += 1

    if (
        run_config
        and run_config.max_llm_calls > 0
        and self._number_of_llm_calls > run_config.max_llm_calls
    ):
      # We only enforce the limit if the limit is a positive number.
      raise LlmCallsLimitExceededError(
          "Max number of llm calls limit of"
          f" `{run_config.max_llm_calls}` exceeded"
      )


class InvocationContext(BaseModel):
  """An invocation context represents the data of a single invocation of an agent.

  An invocation:
    1. Starts with a user message and ends with a final response.
    2. Can contain one or multiple agent calls.
    3. Is handled by runner.run_async().

  An invocation runs an agent until it does not request to transfer to another
  agent.

  An agent call:
    1. Is handled by agent.run().
    2. Ends when agent.run() ends.

  An LLM agent call is an agent with a BaseLLMFlow.
  An LLM agent call can contain one or multiple steps.

  An LLM agent runs steps in a loop until:
    1. A final response is generated.
    2. The agent transfers to another agent.
    3. The end_invocation is set to true by any callbacks or tools.

  A step:
    1. Calls the LLM only once and yields its response.
    2. Calls the tools and yields their responses if requested.

  The summarization of the function response is considered another step, since
  it is another llm call.
  A step ends when it's done calling llm and tools, or if the end_invocation
  is set to true at any time.

  ```
     ┌─────────────────────── invocation ──────────────────────────┐
     ┌──────────── llm_agent_call_1 ────────────┐ ┌─ agent_call_2 ─┐
     ┌──── step_1 ────────┐ ┌───── step_2 ──────┐
     [call_llm] [call_tool] [call_llm] [transfer]
  ```
  """

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra="forbid",
  )
  """The pydantic model config."""

  artifact_service: Optional[BaseArtifactService] = None
  session_service: BaseSessionService
  memory_service: Optional[BaseMemoryService] = None

  invocation_id: str
  """The id of this invocation context. Readonly."""
  branch: Optional[str] = None
  """The branch of the invocation context.

  The format is like agent_1.agent_2.agent_3, where agent_1 is the parent of
  agent_2, and agent_2 is the parent of agent_3.

  Branch is used when multiple sub-agents shouldn't see their peer agents'
  conversation history.
  """
  agent: BaseAgent
  """The current agent of this invocation context. Readonly."""
  user_content: Optional[types.Content] = None
  """The user content that started this invocation. Readonly."""
  session: Session
  """The current session of this invocation context. Readonly."""

  end_invocation: bool = False
  """Whether to end this invocation.

  Set to True in callbacks or tools to terminate this invocation."""

  live_request_queue: Optional[LiveRequestQueue] = None
  """The queue to receive live requests."""

  active_streaming_tools: Optional[dict[str, ActiveStreamingTool]] = None
  """The running streaming tools of this invocation."""

  transcription_cache: Optional[list[TranscriptionEntry]] = None
  """Caches necessary, data audio or contents, that are needed by transcription."""

  run_config: Optional[RunConfig] = None
  """Configurations for live agents under this invocation."""

  _invocation_cost_manager: _InvocationCostManager = _InvocationCostManager()
  """A container to keep track of different kinds of costs incurred as a part
  of this invocation.
  """

  def increment_llm_call_count(
      self,
  ):
    """Tracks number of llm calls made.

    Raises:
      LlmCallsLimitExceededError: If number of llm calls made exceed the set
        threshold.
    """
    self._invocation_cost_manager.increment_and_enforce_llm_calls_limit(
        self.run_config
    )

  @property
  def app_name(self) -> str:
    return self.session.app_name

  @property
  def user_id(self) -> str:
    return self.session.user_id


def new_invocation_context_id() -> str:
  return "e-" + str(uuid.uuid4())



================================================
FILE: src/google/adk/agents/langgraph_agent.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import AsyncGenerator
from typing import Union

from google.genai import types
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from pydantic import ConfigDict
from typing_extensions import override

from ..events.event import Event
from .base_agent import BaseAgent
from .invocation_context import InvocationContext


def _get_last_human_messages(events: list[Event]) -> list[HumanMessage]:
  """Extracts last human messages from given list of events.

  Args:
    events: the list of events

  Returns:
    list of last human messages
  """
  messages = []
  for event in reversed(events):
    if messages and event.author != 'user':
      break
    if event.author == 'user' and event.content and event.content.parts:
      messages.append(HumanMessage(content=event.content.parts[0].text))
  return list(reversed(messages))


class LangGraphAgent(BaseAgent):
  """Currently a concept implementation, supports single and multi-turn."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
  )
  """The pydantic model config."""

  graph: CompiledGraph

  instruction: str = ''

  @override
  async def _run_async_impl(
      self,
      ctx: InvocationContext,
  ) -> AsyncGenerator[Event, None]:

    # Needed for langgraph checkpointer (for subsequent invocations; multi-turn)
    config: RunnableConfig = {'configurable': {'thread_id': ctx.session.id}}

    # Add instruction as SystemMessage if graph state is empty
    current_graph_state = self.graph.get_state(config)
    graph_messages = (
        current_graph_state.values.get('messages', [])
        if current_graph_state.values
        else []
    )
    messages = (
        [SystemMessage(content=self.instruction)]
        if self.instruction and not graph_messages
        else []
    )
    # Add events to messages (evaluating the memory used; parent agent vs checkpointer)
    messages += self._get_messages(ctx.session.events)

    # Use the Runnable
    final_state = self.graph.invoke({'messages': messages}, config)
    result = final_state['messages'][-1].content

    result_event = Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        branch=ctx.branch,
        content=types.Content(
            role='model',
            parts=[types.Part.from_text(text=result)],
        ),
    )
    yield result_event

  def _get_messages(
      self, events: list[Event]
  ) -> list[Union[HumanMessage, AIMessage]]:
    """Extracts messages from given list of events.

    If the developer provides their own memory within langgraph, we return the
    last user messages only. Otherwise, we return all messages between the user
    and the agent.

    Args:
      events: the list of events

    Returns:
      list of messages
    """
    if self.graph.checkpointer:
      return _get_last_human_messages(events)
    else:
      return self._get_conversation_with_agent(events)

  def _get_conversation_with_agent(
      self, events: list[Event]
  ) -> list[Union[HumanMessage, AIMessage]]:
    """Extracts messages from given list of events.

    Args:
      events: the list of events

    Returns:
      list of messages
    """

    messages = []
    for event in events:
      if not event.content or not event.content.parts:
        continue
      if event.author == 'user':
        messages.append(HumanMessage(content=event.content.parts[0].text))
      elif event.author == self.name:
        messages.append(AIMessage(content=event.content.parts[0].text))
    return messages



================================================
FILE: src/google/adk/agents/live_request_queue.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict


class LiveRequest(BaseModel):
  """Request send to live agents."""

  model_config = ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')
  """The pydantic model config."""

  content: Optional[types.Content] = None
  """If set, send the content to the model in turn-by-turn mode."""
  blob: Optional[types.Blob] = None
  """If set, send the blob to the model in realtime mode."""
  close: bool = False
  """If set, close the queue. queue.shutdown() is only supported in Python 3.13+."""


class LiveRequestQueue:
  """Queue used to send LiveRequest in a live(bidirectional streaming) way."""

  def __init__(self):
    # Ensure there's an event loop available in this thread
    try:
      asyncio.get_running_loop()
    except RuntimeError:
      # No running loop, create one
      loop = asyncio.new_event_loop()
      asyncio.set_event_loop(loop)

    # Now create the queue (it will use the event loop we just ensured exists)
    self._queue = asyncio.Queue()

  def close(self):
    self._queue.put_nowait(LiveRequest(close=True))

  def send_content(self, content: types.Content):
    self._queue.put_nowait(LiveRequest(content=content))

  def send_realtime(self, blob: types.Blob):
    self._queue.put_nowait(LiveRequest(blob=blob))

  def send(self, req: LiveRequest):
    self._queue.put_nowait(req)

  async def get(self) -> LiveRequest:
    return await self._queue.get()



================================================
FILE: src/google/adk/agents/llm_agent.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Awaitable, Callable, Literal, Optional, Union

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import override
from typing_extensions import TypeAlias

from ..code_executors.base_code_executor import BaseCodeExecutor
from ..events.event import Event
from ..examples.base_example_provider import BaseExampleProvider
from ..examples.example import Example
from ..flows.llm_flows.auto_flow import AutoFlow
from ..flows.llm_flows.base_llm_flow import BaseLlmFlow
from ..flows.llm_flows.single_flow import SingleFlow
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..planners.base_planner import BasePlanner
from ..tools.base_tool import BaseTool
from ..tools.function_tool import FunctionTool
from ..tools.tool_context import ToolContext
from .base_agent import BaseAgent
from .callback_context import CallbackContext
from .invocation_context import InvocationContext
from .readonly_context import ReadonlyContext

logger = logging.getLogger(__name__)

_SingleBeforeModelCallback: TypeAlias = Callable[
    [CallbackContext, LlmRequest],
    Union[Awaitable[Optional[LlmResponse]], Optional[LlmResponse]],
]

BeforeModelCallback: TypeAlias = Union[
    _SingleBeforeModelCallback,
    list[_SingleBeforeModelCallback],
]

_SingleAfterModelCallback: TypeAlias = Callable[
    [CallbackContext, LlmResponse],
    Union[Awaitable[Optional[LlmResponse]], Optional[LlmResponse]],
]

AfterModelCallback: TypeAlias = Union[
    _SingleAfterModelCallback,
    list[_SingleAfterModelCallback],
]

BeforeToolCallback: TypeAlias = Callable[
    [BaseTool, dict[str, Any], ToolContext],
    Union[Awaitable[Optional[dict]], Optional[dict]],
]
AfterToolCallback: TypeAlias = Callable[
    [BaseTool, dict[str, Any], ToolContext, dict],
    Union[Awaitable[Optional[dict]], Optional[dict]],
]

InstructionProvider: TypeAlias = Callable[[ReadonlyContext], str]

ToolUnion: TypeAlias = Union[Callable, BaseTool]
ExamplesUnion = Union[list[Example], BaseExampleProvider]


def _convert_tool_union_to_tool(
    tool_union: ToolUnion,
) -> BaseTool:
  return (
      tool_union
      if isinstance(tool_union, BaseTool)
      else FunctionTool(tool_union)
  )


class LlmAgent(BaseAgent):
  """LLM-based Agent."""

  model: Union[str, BaseLlm] = ''
  """The model to use for the agent.

  When not set, the agent will inherit the model from its ancestor.
  """

  instruction: Union[str, InstructionProvider] = ''
  """Instructions for the LLM model, guiding the agent's behavior."""

  global_instruction: Union[str, InstructionProvider] = ''
  """Instructions for all the agents in the entire agent tree.

  global_instruction ONLY takes effect in root agent.

  For example: use global_instruction to make all agents have a stable identity
  or personality.
  """

  tools: list[ToolUnion] = Field(default_factory=list)
  """Tools available to this agent."""

  generate_content_config: Optional[types.GenerateContentConfig] = None
  """The additional content generation configurations.

  NOTE: not all fields are usable, e.g. tools must be configured via `tools`,
  thinking_config must be configured via `planner` in LlmAgent.

  For example: use this config to adjust model temperature, configure safety
  settings, etc.
  """

  # LLM-based agent transfer configs - Start
  disallow_transfer_to_parent: bool = False
  """Disallows LLM-controlled transferring to the parent agent."""
  disallow_transfer_to_peers: bool = False
  """Disallows LLM-controlled transferring to the peer agents."""
  # LLM-based agent transfer configs - End

  include_contents: Literal['default', 'none'] = 'default'
  """Whether to include contents in the model request.

  When set to 'none', the model request will not include any contents, such as
  user messages, tool results, etc.
  """

  # Controlled input/output configurations - Start
  input_schema: Optional[type[BaseModel]] = None
  """The input schema when agent is used as a tool."""
  output_schema: Optional[type[BaseModel]] = None
  """The output schema when agent replies.

  NOTE: when this is set, agent can ONLY reply and CANNOT use any tools, such as
  function tools, RAGs, agent transfer, etc.
  """
  output_key: Optional[str] = None
  """The key in session state to store the output of the agent.

  Typically use cases:
  - Extracts agent reply for later use, such as in tools, callbacks, etc.
  - Connects agents to coordinate with each other.
  """
  # Controlled input/output configurations - End

  # Advance features - Start
  planner: Optional[BasePlanner] = None
  """Instructs the agent to make a plan and execute it step by step.

  NOTE: to use model's built-in thinking features, set the `thinking_config`
  field in `google.adk.planners.built_in_planner`.

  """

  code_executor: Optional[BaseCodeExecutor] = None
  """Allow agent to execute code blocks from model responses using the provided
  CodeExecutor.

  Check out available code executions in `google.adk.code_executor` package.

  NOTE: to use model's built-in code executor, don't set this field, add
  `google.adk.tools.built_in_code_execution` to tools instead.
  """
  # Advance features - End

  # TODO: remove below fields after migration. - Start
  # These fields are added back for easier migration.
  examples: Optional[ExamplesUnion] = None
  # TODO: remove above fields after migration. - End

  # Callbacks - Start
  before_model_callback: Optional[BeforeModelCallback] = None
  """Callback or list of callbacks to be called before calling the LLM.

  When a list of callbacks is provided, the callbacks will be called in the
  order they are listed until a callback does not return None.

  Args:
    callback_context: CallbackContext,
    llm_request: LlmRequest, The raw model request. Callback can mutate the
    request.

  Returns:
    The content to return to the user. When present, the model call will be
    skipped and the provided content will be returned to user.
  """
  after_model_callback: Optional[AfterModelCallback] = None
  """Callback or list of callbacks to be called after calling the LLM.

  When a list of callbacks is provided, the callbacks will be called in the
  order they are listed until a callback does not return None.

  Args:
    callback_context: CallbackContext,
    llm_response: LlmResponse, the actual model response.

  Returns:
    The content to return to the user. When present, the actual model response
    will be ignored and the provided content will be returned to user.
  """
  before_tool_callback: Optional[BeforeToolCallback] = None
  """Called before the tool is called.

  Args:
    tool: The tool to be called.
    args: The arguments to the tool.
    tool_context: ToolContext,

  Returns:
    The tool response. When present, the returned tool response will be used and
    the framework will skip calling the actual tool.
  """
  after_tool_callback: Optional[AfterToolCallback] = None
  """Called after the tool is called.

  Args:
    tool: The tool to be called.
    args: The arguments to the tool.
    tool_context: ToolContext,
    tool_response: The response from the tool.

  Returns:
    When present, the returned dict will be used as tool result.
  """
  # Callbacks - End

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    async for event in self._llm_flow.run_async(ctx):
      self.__maybe_save_output_to_state(event)
      yield event

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    async for event in self._llm_flow.run_live(ctx):
      self.__maybe_save_output_to_state(event)
      yield event
    if ctx.end_invocation:
      return

  @property
  def canonical_model(self) -> BaseLlm:
    """The resolved self.model field as BaseLlm.

    This method is only for use by Agent Development Kit.
    """
    if isinstance(self.model, BaseLlm):
      return self.model
    elif self.model:  # model is non-empty str
      return LLMRegistry.new_llm(self.model)
    else:  # find model from ancestors.
      ancestor_agent = self.parent_agent
      while ancestor_agent is not None:
        if isinstance(ancestor_agent, LlmAgent):
          return ancestor_agent.canonical_model
        ancestor_agent = ancestor_agent.parent_agent
      raise ValueError(f'No model found for {self.name}.')

  def canonical_instruction(self, ctx: ReadonlyContext) -> str:
    """The resolved self.instruction field to construct instruction for this agent.

    This method is only for use by Agent Development Kit.
    """
    if isinstance(self.instruction, str):
      return self.instruction
    else:
      return self.instruction(ctx)

  def canonical_global_instruction(self, ctx: ReadonlyContext) -> str:
    """The resolved self.instruction field to construct global instruction.

    This method is only for use by Agent Development Kit.
    """
    if isinstance(self.global_instruction, str):
      return self.global_instruction
    else:
      return self.global_instruction(ctx)

  @property
  def canonical_tools(self) -> list[BaseTool]:
    """The resolved self.tools field as a list of BaseTool.

    This method is only for use by Agent Development Kit.
    """
    return [_convert_tool_union_to_tool(tool) for tool in self.tools]

  @property
  def canonical_before_model_callbacks(
      self,
  ) -> list[_SingleBeforeModelCallback]:
    """The resolved self.before_model_callback field as a list of _SingleBeforeModelCallback.

    This method is only for use by Agent Development Kit.
    """
    if not self.before_model_callback:
      return []
    if isinstance(self.before_model_callback, list):
      return self.before_model_callback
    return [self.before_model_callback]

  @property
  def canonical_after_model_callbacks(self) -> list[_SingleAfterModelCallback]:
    """The resolved self.after_model_callback field as a list of _SingleAfterModelCallback.

    This method is only for use by Agent Development Kit.
    """
    if not self.after_model_callback:
      return []
    if isinstance(self.after_model_callback, list):
      return self.after_model_callback
    return [self.after_model_callback]

  @property
  def _llm_flow(self) -> BaseLlmFlow:
    if (
        self.disallow_transfer_to_parent
        and self.disallow_transfer_to_peers
        and not self.sub_agents
    ):
      return SingleFlow()
    else:
      return AutoFlow()

  def __maybe_save_output_to_state(self, event: Event):
    """Saves the model output to state if needed."""
    if (
        self.output_key
        and event.is_final_response()
        and event.content
        and event.content.parts
    ):
      result = ''.join(
          [part.text if part.text else '' for part in event.content.parts]
      )
      if self.output_schema:
        result = self.output_schema.model_validate_json(result).model_dump(
            exclude_none=True
        )
      event.actions.state_delta[self.output_key] = result

  @model_validator(mode='after')
  def __model_validator_after(self) -> LlmAgent:
    self.__check_output_schema()
    return self

  def __check_output_schema(self):
    if not self.output_schema:
      return

    if (
        not self.disallow_transfer_to_parent
        or not self.disallow_transfer_to_peers
    ):
      logger.warning(
          'Invalid config for agent %s: output_schema cannot co-exist with'
          ' agent transfer configurations. Setting'
          ' disallow_transfer_to_parent=True, disallow_transfer_to_peers=True',
          self.name,
      )
      self.disallow_transfer_to_parent = True
      self.disallow_transfer_to_peers = True

    if self.sub_agents:
      raise ValueError(
          f'Invalid config for agent {self.name}: if output_schema is set,'
          ' sub_agents must be empty to disable agent transfer.'
      )

    if self.tools:
      raise ValueError(
          f'Invalid config for agent {self.name}: if output_schema is set,'
          ' tools must be empty'
      )

  @field_validator('generate_content_config', mode='after')
  @classmethod
  def __validate_generate_content_config(
      cls, generate_content_config: Optional[types.GenerateContentConfig]
  ) -> types.GenerateContentConfig:
    if not generate_content_config:
      return types.GenerateContentConfig()
    if generate_content_config.thinking_config:
      raise ValueError('Thinking config should be set via LlmAgent.planner.')
    if generate_content_config.tools:
      raise ValueError('All tools must be set via LlmAgent.tools.')
    if generate_content_config.system_instruction:
      raise ValueError(
          'System instruction must be set via LlmAgent.instruction.'
      )
    if generate_content_config.response_schema:
      raise ValueError(
          'Response schema must be set via LlmAgent.output_schema.'
      )
    return generate_content_config


Agent: TypeAlias = LlmAgent



================================================
FILE: src/google/adk/agents/loop_agent.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loop agent implementation."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import Optional

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from .base_agent import BaseAgent


class LoopAgent(BaseAgent):
  """A shell agent that run its sub-agents in a loop.

  When sub-agent generates an event with escalate or max_iterations are
  reached, the loop agent will stop.
  """

  max_iterations: Optional[int] = None
  """The maximum number of iterations to run the loop agent.

  If not set, the loop agent will run indefinitely until a sub-agent
  escalates.
  """

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    times_looped = 0
    while not self.max_iterations or times_looped < self.max_iterations:
      for sub_agent in self.sub_agents:
        async for event in sub_agent.run_async(ctx):
          yield event
          if event.actions.escalate:
            return
      times_looped += 1
    return

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    raise NotImplementedError('The behavior for run_live is not defined yet.')
    yield  # AsyncGenerator requires having at least one yield statement



================================================
FILE: src/google/adk/agents/parallel_agent.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallel agent implementation."""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from .base_agent import BaseAgent


def _set_branch_for_current_agent(
    current_agent: BaseAgent, invocation_context: InvocationContext
):
  invocation_context.branch = (
      f"{invocation_context.branch}.{current_agent.name}"
      if invocation_context.branch
      else current_agent.name
  )


async def _merge_agent_run(
    agent_runs: list[AsyncGenerator[Event, None]],
) -> AsyncGenerator[Event, None]:
  """Merges the agent run event generator.

  This implementation guarantees for each agent, it won't move on until the
  generated event is processed by upstream runner.

  Args:
      agent_runs: A list of async generators that yield events from each agent.

  Yields:
      Event: The next event from the merged generator.
  """
  tasks = [
      asyncio.create_task(events_for_one_agent.__anext__())
      for events_for_one_agent in agent_runs
  ]
  pending_tasks = set(tasks)

  while pending_tasks:
    done, pending_tasks = await asyncio.wait(
        pending_tasks, return_when=asyncio.FIRST_COMPLETED
    )
    for task in done:
      try:
        yield task.result()

        # Find the generator that produced this event and move it on.
        for i, original_task in enumerate(tasks):
          if task == original_task:
            new_task = asyncio.create_task(agent_runs[i].__anext__())
            tasks[i] = new_task
            pending_tasks.add(new_task)
            break  # stop iterating once found

      except StopAsyncIteration:
        continue


class ParallelAgent(BaseAgent):
  """A shell agent that run its sub-agents in parallel in isolated manner.

  This approach is beneficial for scenarios requiring multiple perspectives or
  attempts on a single task, such as:

  - Running different algorithms simultaneously.
  - Generating multiple responses for review by a subsequent evaluation agent.
  """

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    _set_branch_for_current_agent(self, ctx)
    agent_runs = [agent.run_async(ctx) for agent in self.sub_agents]
    async for event in _merge_agent_run(agent_runs):
      yield event



================================================
FILE: src/google/adk/agents/readonly_context.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from types import MappingProxyType
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .invocation_context import InvocationContext


class ReadonlyContext:

  def __init__(
      self,
      invocation_context: InvocationContext,
  ) -> None:
    self._invocation_context = invocation_context

  @property
  def invocation_id(self) -> str:
    """The current invocation id."""
    return self._invocation_context.invocation_id

  @property
  def agent_name(self) -> str:
    """The name of the agent that is currently running."""
    return self._invocation_context.agent.name

  @property
  def state(self) -> MappingProxyType[str, Any]:
    """The state of the current session. READONLY field."""
    return MappingProxyType(self._invocation_context.session.state)



================================================
FILE: src/google/adk/agents/remote_agent.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import AsyncGenerator

from pydantic import Field
import requests
from typing_extensions import override

from ..events.event import Event
from .base_agent import BaseAgent
from .invocation_context import InvocationContext


class RemoteAgent(BaseAgent):
  """Experimental, do not use."""

  url: str

  sub_agents: list[BaseAgent] = Field(
      default_factory=list, init=False, frozen=True
  )
  """Sub-agent is disabled in RemoteAgent."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    data = {
        'invocation_id': ctx.invocation_id,
        'session': ctx.session.model_dump(exclude_none=True),
    }
    events = requests.post(self.url, data=json.dumps(data), timeout=120)
    events.raise_for_status()
    for event in events.json():
      e = Event.model_validate(event)
      e.author = self.name
      yield e



================================================
FILE: src/google/adk/agents/run_config.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import logging
import sys
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
  NONE = None
  SSE = 'sse'
  BIDI = 'bidi'


class RunConfig(BaseModel):
  """Configs for runtime behavior of agents."""

  model_config = ConfigDict(
      extra='forbid',
  )
  """The pydantic model config."""

  speech_config: Optional[types.SpeechConfig] = None
  """Speech configuration for the live agent."""

  response_modalities: Optional[list[str]] = None
  """The output modalities. If not set, it's default to AUDIO."""

  save_input_blobs_as_artifacts: bool = False
  """Whether or not to save the input blobs as artifacts."""

  support_cfc: bool = False
  """
  Whether to support CFC (Compositional Function Calling). Only applicable for
  StreamingMode.SSE. If it's true. the LIVE API will be invoked. Since only LIVE
  API supports CFC

  .. warning::
      This feature is **experimental** and its API or behavior may change
      in future releases.
  """

  streaming_mode: StreamingMode = StreamingMode.NONE
  """Streaming mode, None or StreamingMode.SSE or StreamingMode.BIDI."""

  output_audio_transcription: Optional[types.AudioTranscriptionConfig] = None
  """Output transcription for live agents with audio response."""

  input_audio_transcription: Optional[types.AudioTranscriptionConfig] = None
  """Input transcription for live agents with audio input from user."""

  max_llm_calls: int = 500
  """
  A limit on the total number of llm calls for a given run.

  Valid Values:
    - More than 0 and less than sys.maxsize: The bound on the number of llm
      calls is enforced, if the value is set in this range.
    - Less than or equal to 0: This allows for unbounded number of llm calls.
  """

  @field_validator('max_llm_calls', mode='after')
  @classmethod
  def validate_max_llm_calls(cls, value: int) -> int:
    if value == sys.maxsize:
      raise ValueError(f'max_llm_calls should be less than {sys.maxsize}.')
    elif value <= 0:
      logger.warning(
          'max_llm_calls is less than or equal to 0. This will result in'
          ' no enforcement on total number of llm calls that will be made for a'
          ' run. This may not be ideal, as this could result in a never'
          ' ending communication between the model and the agent in certain'
          ' cases.',
      )

    return value



================================================
FILE: src/google/adk/agents/sequential_agent.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequential agent implementation."""

from __future__ import annotations

from typing import AsyncGenerator

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from .base_agent import BaseAgent


class SequentialAgent(BaseAgent):
  """A shell agent that run its sub-agents in sequence."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for sub_agent in self.sub_agents:
      async for event in sub_agent.run_async(ctx):
        yield event

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for sub_agent in self.sub_agents:
      async for event in sub_agent.run_live(ctx):
        yield event



================================================
FILE: src/google/adk/agents/transcription_entry.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict


class TranscriptionEntry(BaseModel):
  """Store the data that can be used for transcription."""

  model_config = ConfigDict(
      arbitrary_types_allowed=True,
      extra='forbid',
  )
  """The pydantic model config."""

  role: str
  """The role that created this data, typically "user" or "model"""

  data: Union[types.Blob, types.Content]
  """The data that can be used for transcription"""



================================================
FILE: src/google/adk/artifacts/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base_artifact_service import BaseArtifactService
from .gcs_artifact_service import GcsArtifactService
from .in_memory_artifact_service import InMemoryArtifactService

__all__ = [
    'BaseArtifactService',
    'GcsArtifactService',
    'InMemoryArtifactService',
]



================================================
FILE: src/google/adk/artifacts/base_artifact_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC
from abc import abstractmethod
from typing import Optional

from google.genai import types


class BaseArtifactService(ABC):
  """Abstract base class for artifact services."""

  @abstractmethod
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      artifact: types.Part,
  ) -> int:
    """Saves an artifact to the artifact service storage.

    The artifact is a file identified by the app name, user ID, session ID, and
    filename. After saving the artifact, a revision ID is returned to identify
    the artifact version.

    Args:
      app_name: The app name.
      user_id: The user ID.
      session_id: The session ID.
      filename: The filename of the artifact.
      artifact: The artifact to save.

    Returns:
      The revision ID. The first version of the artifact has a revision ID of 0.
      This is incremented by 1 after each successful save.
    """

  @abstractmethod
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    """Gets an artifact from the artifact service storage.

    The artifact is a file identified by the app name, user ID, session ID, and
    filename.

    Args:
      app_name: The app name.
      user_id: The user ID.
      session_id: The session ID.
      filename: The filename of the artifact.
      version: The version of the artifact. If None, the latest version will be
        returned.

    Returns:
      The artifact or None if not found.
    """

  @abstractmethod
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> list[str]:
    """Lists all the artifact filenames within a session.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.

    Returns:
        A list of all artifact filenames within a session.
    """

  @abstractmethod
  async def delete_artifact(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> None:
    """Deletes an artifact.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.
    """

  @abstractmethod
  async def list_versions(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> list[int]:
    """Lists all versions of an artifact.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.

    Returns:
        A list of all available versions of the artifact.
    """



================================================
FILE: src/google/adk/artifacts/gcs_artifact_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An artifact service implementation using Google Cloud Storage (GCS)."""

import logging
from typing import Optional

from google.cloud import storage
from google.genai import types
from typing_extensions import override

from .base_artifact_service import BaseArtifactService

logger = logging.getLogger(__name__)


class GcsArtifactService(BaseArtifactService):
  """An artifact service implementation using Google Cloud Storage (GCS)."""

  def __init__(self, bucket_name: str, **kwargs):
    """Initializes the GcsArtifactService.

    Args:
        bucket_name: The name of the bucket to use.
        **kwargs: Keyword arguments to pass to the Google Cloud Storage client.
    """
    self.bucket_name = bucket_name
    self.storage_client = storage.Client(**kwargs)
    self.bucket = self.storage_client.bucket(self.bucket_name)

  def _file_has_user_namespace(self, filename: str) -> bool:
    """Checks if the filename has a user namespace.

    Args:
        filename: The filename to check.

    Returns:
        True if the filename has a user namespace (starts with "user:"),
        False otherwise.
    """
    return filename.startswith("user:")

  def _get_blob_name(
      self,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: int,
  ) -> str:
    """Constructs the blob name in GCS.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.
        version: The version of the artifact.

    Returns:
        The constructed blob name in GCS.
    """
    if self._file_has_user_namespace(filename):
      return f"{app_name}/{user_id}/user/{filename}/{version}"
    return f"{app_name}/{user_id}/{session_id}/{filename}/{version}"

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      artifact: types.Part,
  ) -> int:
    versions = await self.list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    version = 0 if not versions else max(versions) + 1

    blob_name = self._get_blob_name(
        app_name, user_id, session_id, filename, version
    )
    blob = self.bucket.blob(blob_name)

    blob.upload_from_string(
        data=artifact.inline_data.data,
        content_type=artifact.inline_data.mime_type,
    )

    return version

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    if version is None:
      versions = await self.list_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=filename,
      )
      if not versions:
        return None
      version = max(versions)

    blob_name = self._get_blob_name(
        app_name, user_id, session_id, filename, version
    )
    blob = self.bucket.blob(blob_name)

    artifact_bytes = blob.download_as_bytes()
    if not artifact_bytes:
      return None
    artifact = types.Part.from_bytes(
        data=artifact_bytes, mime_type=blob.content_type
    )
    return artifact

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> list[str]:
    filenames = set()

    session_prefix = f"{app_name}/{user_id}/{session_id}/"
    session_blobs = self.storage_client.list_blobs(
        self.bucket, prefix=session_prefix
    )
    for blob in session_blobs:
      _, _, _, filename, _ = blob.name.split("/")
      filenames.add(filename)

    user_namespace_prefix = f"{app_name}/{user_id}/user/"
    user_namespace_blobs = self.storage_client.list_blobs(
        self.bucket, prefix=user_namespace_prefix
    )
    for blob in user_namespace_blobs:
      _, _, _, filename, _ = blob.name.split("/")
      filenames.add(filename)

    return sorted(list(filenames))

  @override
  async def delete_artifact(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> None:
    versions = await self.list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    for version in versions:
      blob_name = self._get_blob_name(
          app_name, user_id, session_id, filename, version
      )
      blob = self.bucket.blob(blob_name)
      blob.delete()
    return

  @override
  async def list_versions(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> list[int]:
    prefix = self._get_blob_name(app_name, user_id, session_id, filename, "")
    blobs = self.storage_client.list_blobs(self.bucket, prefix=prefix)
    versions = []
    for blob in blobs:
      _, _, _, _, version = blob.name.split("/")
      versions.append(int(version))
    return versions



================================================
FILE: src/google/adk/artifacts/in_memory_artifact_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An in-memory implementation of the artifact service."""

import logging
from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from .base_artifact_service import BaseArtifactService

logger = logging.getLogger(__name__)


class InMemoryArtifactService(BaseArtifactService, BaseModel):
  """An in-memory implementation of the artifact service."""

  artifacts: dict[str, list[types.Part]] = Field(default_factory=dict)

  def _file_has_user_namespace(self, filename: str) -> bool:
    """Checks if the filename has a user namespace.

    Args:
        filename: The filename to check.

    Returns:
        True if the filename has a user namespace (starts with "user:"),
        False otherwise.
    """
    return filename.startswith("user:")

  def _artifact_path(
      self, app_name: str, user_id: str, session_id: str, filename: str
  ) -> str:
    """Constructs the artifact path.

    Args:
        app_name: The name of the application.
        user_id: The ID of the user.
        session_id: The ID of the session.
        filename: The name of the artifact file.

    Returns:
        The constructed artifact path.
    """
    if self._file_has_user_namespace(filename):
      return f"{app_name}/{user_id}/user/{filename}"
    return f"{app_name}/{user_id}/{session_id}/{filename}"

  @override
  async def save_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      artifact: types.Part,
  ) -> int:
    path = self._artifact_path(app_name, user_id, session_id, filename)
    if path not in self.artifacts:
      self.artifacts[path] = []
    version = len(self.artifacts[path])
    self.artifacts[path].append(artifact)
    return version

  @override
  async def load_artifact(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      filename: str,
      version: Optional[int] = None,
  ) -> Optional[types.Part]:
    path = self._artifact_path(app_name, user_id, session_id, filename)
    versions = self.artifacts.get(path)
    if not versions:
      return None
    if version is None:
      version = -1
    return versions[version]

  @override
  async def list_artifact_keys(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> list[str]:
    session_prefix = f"{app_name}/{user_id}/{session_id}/"
    usernamespace_prefix = f"{app_name}/{user_id}/user/"
    filenames = []
    for path in self.artifacts:
      if path.startswith(session_prefix):
        filename = path.removeprefix(session_prefix)
        filenames.append(filename)
      elif path.startswith(usernamespace_prefix):
        filename = path.removeprefix(usernamespace_prefix)
        filenames.append(filename)
    return sorted(filenames)

  @override
  async def delete_artifact(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> None:
    path = self._artifact_path(app_name, user_id, session_id, filename)
    if not self.artifacts.get(path):
      return None
    self.artifacts.pop(path, None)

  @override
  async def list_versions(
      self, *, app_name: str, user_id: str, session_id: str, filename: str
  ) -> list[int]:
    path = self._artifact_path(app_name, user_id, session_id, filename)
    versions = self.artifacts.get(path)
    if not versions:
      return []
    return list(range(len(versions)))



================================================
FILE: src/google/adk/auth/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_credential import OAuth2Auth
from .auth_handler import AuthHandler
from .auth_schemes import AuthScheme
from .auth_schemes import AuthSchemeType
from .auth_schemes import OpenIdConnectWithConfig
from .auth_tool import AuthConfig



================================================
FILE: src/google/adk/auth/auth_credential.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class BaseModelWithConfig(BaseModel):
  model_config = ConfigDict(extra="allow")
  """The pydantic model config."""


class HttpCredentials(BaseModelWithConfig):
  """Represents the secret token value for HTTP authentication, like user name, password, oauth token, etc."""

  username: Optional[str] = None
  password: Optional[str] = None
  token: Optional[str] = None

  @classmethod
  def model_validate(cls, data: Dict[str, Any]) -> "HttpCredentials":
    return cls(
        username=data.get("username"),
        password=data.get("password"),
        token=data.get("token"),
    )


class HttpAuth(BaseModelWithConfig):
  """The credentials and metadata for HTTP authentication."""

  # The name of the HTTP Authorization scheme to be used in the Authorization
  # header as defined in RFC7235. The values used SHOULD be registered in the
  # IANA Authentication Scheme registry.
  # Examples: 'basic', 'bearer'
  scheme: str
  credentials: HttpCredentials


class OAuth2Auth(BaseModelWithConfig):
  """Represents credential value and its metadata for a OAuth2 credential."""

  client_id: Optional[str] = None
  client_secret: Optional[str] = None
  # tool or adk can generate the auth_uri with the state info thus client
  # can verify the state
  auth_uri: Optional[str] = None
  state: Optional[str] = None
  # tool or adk can decide the redirect_uri if they don't want client to decide
  redirect_uri: Optional[str] = None
  auth_response_uri: Optional[str] = None
  auth_code: Optional[str] = None
  access_token: Optional[str] = None
  refresh_token: Optional[str] = None


class ServiceAccountCredential(BaseModelWithConfig):
  """Represents Google Service Account configuration.

  Attributes:
    type: The type should be "service_account".
    project_id: The project ID.
    private_key_id: The ID of the private key.
    private_key: The private key.
    client_email: The client email.
    client_id: The client ID.
    auth_uri: The authorization URI.
    token_uri: The token URI.
    auth_provider_x509_cert_url: URL for auth provider's X.509 cert.
    client_x509_cert_url: URL for the client's X.509 cert.
    universe_domain: The universe domain.

  Example:

      config = ServiceAccountCredential(
          type_="service_account",
          project_id="your_project_id",
          private_key_id="your_private_key_id",
          private_key="-----BEGIN PRIVATE KEY-----...",
          client_email="...@....iam.gserviceaccount.com",
          client_id="your_client_id",
          auth_uri="https://accounts.google.com/o/oauth2/auth",
          token_uri="https://oauth2.googleapis.com/token",
          auth_provider_x509_cert_url="https://www.googleapis.com/oauth2/v1/certs",
          client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/...",
          universe_domain="googleapis.com"
      )


      config = ServiceAccountConfig.model_construct(**{
          ...service account config dict
      })
  """

  type_: str = Field("", alias="type")
  project_id: str
  private_key_id: str
  private_key: str
  client_email: str
  client_id: str
  auth_uri: str
  token_uri: str
  auth_provider_x509_cert_url: str
  client_x509_cert_url: str
  universe_domain: str


class ServiceAccount(BaseModelWithConfig):
  """Represents Google Service Account configuration."""

  service_account_credential: Optional[ServiceAccountCredential] = None
  scopes: List[str]
  use_default_credential: Optional[bool] = False


class AuthCredentialTypes(str, Enum):
  """Represents the type of authentication credential."""

  # API Key credential:
  # https://swagger.io/docs/specification/v3_0/authentication/api-keys/
  API_KEY = "apiKey"

  # Credentials for HTTP Auth schemes:
  # https://www.iana.org/assignments/http-authschemes/http-authschemes.xhtml
  HTTP = "http"

  # OAuth2 credentials:
  # https://swagger.io/docs/specification/v3_0/authentication/oauth2/
  OAUTH2 = "oauth2"

  # OpenID Connect credentials:
  # https://swagger.io/docs/specification/v3_0/authentication/openid-connect-discovery/
  OPEN_ID_CONNECT = "openIdConnect"

  # Service Account credentials:
  # https://cloud.google.com/iam/docs/service-account-creds
  SERVICE_ACCOUNT = "serviceAccount"


class AuthCredential(BaseModelWithConfig):
  """Data class representing an authentication credential.

  To exchange for the actual credential, please use
  CredentialExchanger.exchange_credential().

  Examples: API Key Auth
  AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY,
      api_key="1234",
  )

  Example: HTTP Auth
  AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="basic",
          credentials=HttpCredentials(username="user", password="password"),
      ),
  )

  Example: OAuth2 Bearer Token in HTTP Header
  AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer",
          credentials=HttpCredentials(token="eyAkaknabna...."),
      ),
  )

  Example: OAuth2 Auth with Authorization Code Flow
  AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="1234",
          client_secret="secret",
      ),
  )

  Example: OpenID Connect Auth
  AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id="1234",
          client_secret="secret",
          redirect_uri="https://example.com",
          scopes=["scope1", "scope2"],
      ),
  )

  Example: Auth with resource reference
  AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY,
      resource_ref="projects/1234/locations/us-central1/resources/resource1",
  )
  """

  auth_type: AuthCredentialTypes
  # Resource reference for the credential.
  # This will be supported in the future.
  resource_ref: Optional[str] = None

  api_key: Optional[str] = None
  http: Optional[HttpAuth] = None
  service_account: Optional[ServiceAccount] = None
  oauth2: Optional[OAuth2Auth] = None



================================================
FILE: src/google/adk/auth/auth_handler.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import SecurityBase

from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_credential import OAuth2Auth
from .auth_schemes import AuthSchemeType
from .auth_schemes import OAuthGrantType
from .auth_schemes import OpenIdConnectWithConfig
from .auth_tool import AuthConfig

if TYPE_CHECKING:
  from ..sessions.state import State

try:
  from authlib.integrations.requests_client import OAuth2Session

  SUPPORT_TOKEN_EXCHANGE = True
except ImportError:
  SUPPORT_TOKEN_EXCHANGE = False


class AuthHandler:

  def __init__(self, auth_config: AuthConfig):
    self.auth_config = auth_config

  def exchange_auth_token(
      self,
  ) -> AuthCredential:
    """Generates an auth token from the authorization response.

    Returns:
        An AuthCredential object containing the access token.

    Raises:
        ValueError: If the token endpoint is not configured in the auth
            scheme.
        AuthCredentialMissingError: If the access token cannot be retrieved
            from the token endpoint.
    """
    auth_scheme = self.auth_config.auth_scheme
    auth_credential = self.auth_config.exchanged_auth_credential
    if not SUPPORT_TOKEN_EXCHANGE:
      return auth_credential
    if isinstance(auth_scheme, OpenIdConnectWithConfig):
      if not hasattr(auth_scheme, "token_endpoint"):
        return self.auth_config.exchanged_auth_credential
      token_endpoint = auth_scheme.token_endpoint
      scopes = auth_scheme.scopes
    elif isinstance(auth_scheme, OAuth2):
      if (
          not auth_scheme.flows.authorizationCode
          or not auth_scheme.flows.authorizationCode.tokenUrl
      ):
        return self.auth_config.exchanged_auth_credential
      token_endpoint = auth_scheme.flows.authorizationCode.tokenUrl
      scopes = list(auth_scheme.flows.authorizationCode.scopes.keys())
    else:
      return self.auth_config.exchanged_auth_credential

    if (
        not auth_credential
        or not auth_credential.oauth2
        or not auth_credential.oauth2.client_id
        or not auth_credential.oauth2.client_secret
        or auth_credential.oauth2.access_token
        or auth_credential.oauth2.refresh_token
    ):
      return self.auth_config.exchanged_auth_credential

    client = OAuth2Session(
        auth_credential.oauth2.client_id,
        auth_credential.oauth2.client_secret,
        scope=" ".join(scopes),
        redirect_uri=auth_credential.oauth2.redirect_uri,
        state=auth_credential.oauth2.state,
    )
    tokens = client.fetch_token(
        token_endpoint,
        authorization_response=auth_credential.oauth2.auth_response_uri,
        code=auth_credential.oauth2.auth_code,
        grant_type=OAuthGrantType.AUTHORIZATION_CODE,
    )

    updated_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
        ),
    )
    return updated_credential

  def parse_and_store_auth_response(self, state: State) -> None:

    credential_key = self.get_credential_key()

    state[credential_key] = self.auth_config.exchanged_auth_credential
    if not isinstance(
        self.auth_config.auth_scheme, SecurityBase
    ) or self.auth_config.auth_scheme.type_ not in (
        AuthSchemeType.oauth2,
        AuthSchemeType.openIdConnect,
    ):
      return

    state[credential_key] = self.exchange_auth_token()

  def _validate(self) -> None:
    if not self.auth_scheme:
      raise ValueError("auth_scheme is empty.")

  def get_auth_response(self, state: State) -> AuthCredential:
    credential_key = self.get_credential_key()
    return state.get(credential_key, None)

  def generate_auth_request(self) -> AuthConfig:
    if not isinstance(
        self.auth_config.auth_scheme, SecurityBase
    ) or self.auth_config.auth_scheme.type_ not in (
        AuthSchemeType.oauth2,
        AuthSchemeType.openIdConnect,
    ):
      return self.auth_config.model_copy(deep=True)

    # auth_uri already in exchanged credential
    if (
        self.auth_config.exchanged_auth_credential
        and self.auth_config.exchanged_auth_credential.oauth2
        and self.auth_config.exchanged_auth_credential.oauth2.auth_uri
    ):
      return self.auth_config.model_copy(deep=True)

    # Check if raw_auth_credential exists
    if not self.auth_config.raw_auth_credential:
      raise ValueError(
          f"Auth Scheme {self.auth_config.auth_scheme.type_} requires"
          " auth_credential."
      )

    # Check if oauth2 exists in raw_auth_credential
    if not self.auth_config.raw_auth_credential.oauth2:
      raise ValueError(
          f"Auth Scheme {self.auth_config.auth_scheme.type_} requires oauth2 in"
          " auth_credential."
      )

    # auth_uri in raw credential
    if self.auth_config.raw_auth_credential.oauth2.auth_uri:
      return AuthConfig(
          auth_scheme=self.auth_config.auth_scheme,
          raw_auth_credential=self.auth_config.raw_auth_credential,
          exchanged_auth_credential=self.auth_config.raw_auth_credential.model_copy(
              deep=True
          ),
      )

    # Check for client_id and client_secret
    if (
        not self.auth_config.raw_auth_credential.oauth2.client_id
        or not self.auth_config.raw_auth_credential.oauth2.client_secret
    ):
      raise ValueError(
          f"Auth Scheme {self.auth_config.auth_scheme.type_} requires both"
          " client_id and client_secret in auth_credential.oauth2."
      )

    # Generate new auth URI
    exchanged_credential = self.generate_auth_uri()
    return AuthConfig(
        auth_scheme=self.auth_config.auth_scheme,
        raw_auth_credential=self.auth_config.raw_auth_credential,
        exchanged_auth_credential=exchanged_credential,
    )

  def get_credential_key(self) -> str:
    """Generates a unique key for the given auth scheme and credential."""
    auth_scheme = self.auth_config.auth_scheme
    auth_credential = self.auth_config.raw_auth_credential
    if auth_scheme.model_extra:
      auth_scheme = auth_scheme.model_copy(deep=True)
      auth_scheme.model_extra.clear()
    scheme_name = (
        f"{auth_scheme.type_.name}_{hash(auth_scheme.model_dump_json())}"
        if auth_scheme
        else ""
    )
    if auth_credential.model_extra:
      auth_credential = auth_credential.model_copy(deep=True)
      auth_credential.model_extra.clear()
    credential_name = (
        f"{auth_credential.auth_type.value}_{hash(auth_credential.model_dump_json())}"
        if auth_credential
        else ""
    )

    return f"temp:adk_{scheme_name}_{credential_name}"

  def generate_auth_uri(
      self,
  ) -> AuthCredential:
    """Generates an response containing the auth uri for user to sign in.

    Returns:
        An AuthCredential object containing the auth URI and state.

    Raises:
        ValueError: If the authorization endpoint is not configured in the auth
            scheme.
    """
    auth_scheme = self.auth_config.auth_scheme
    auth_credential = self.auth_config.raw_auth_credential

    if isinstance(auth_scheme, OpenIdConnectWithConfig):
      authorization_endpoint = auth_scheme.authorization_endpoint
      scopes = auth_scheme.scopes
    else:
      authorization_endpoint = (
          auth_scheme.flows.implicit
          and auth_scheme.flows.implicit.authorizationUrl
          or auth_scheme.flows.authorizationCode
          and auth_scheme.flows.authorizationCode.authorizationUrl
          or auth_scheme.flows.clientCredentials
          and auth_scheme.flows.clientCredentials.tokenUrl
          or auth_scheme.flows.password
          and auth_scheme.flows.password.tokenUrl
      )
      scopes = (
          auth_scheme.flows.implicit
          and auth_scheme.flows.implicit.scopes
          or auth_scheme.flows.authorizationCode
          and auth_scheme.flows.authorizationCode.scopes
          or auth_scheme.flows.clientCredentials
          and auth_scheme.flows.clientCredentials.scopes
          or auth_scheme.flows.password
          and auth_scheme.flows.password.scopes
      )
      scopes = list(scopes.keys())

    client = OAuth2Session(
        auth_credential.oauth2.client_id,
        auth_credential.oauth2.client_secret,
        scope=" ".join(scopes),
        redirect_uri=auth_credential.oauth2.redirect_uri,
    )
    uri, state = client.create_authorization_url(
        url=authorization_endpoint, access_type="offline", prompt="consent"
    )
    exchanged_auth_credential = auth_credential.model_copy(deep=True)
    exchanged_auth_credential.oauth2.auth_uri = uri
    exchanged_auth_credential.oauth2.state = state

    return exchanged_auth_credential



================================================
FILE: src/google/adk/auth/auth_preprocessor.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import AsyncGenerator
from typing import TYPE_CHECKING

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from ..flows.llm_flows import functions
from ..flows.llm_flows._base_llm_processor import BaseLlmRequestProcessor
from ..flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from ..models.llm_request import LlmRequest
from .auth_handler import AuthHandler
from .auth_tool import AuthConfig
from .auth_tool import AuthToolArguments

if TYPE_CHECKING:
  from ..agents.llm_agent import LlmAgent


class _AuthLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles auth information to build the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ..agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return
    events = invocation_context.session.events
    if not events:
      return

    request_euc_function_call_ids = set()
    for k in range(len(events) - 1, -1, -1):
      event = events[k]
      # look for first event authored by user
      if not event.author or event.author != 'user':
        continue
      responses = event.get_function_responses()
      if not responses:
        return

      for function_call_response in responses:
        if function_call_response.name != REQUEST_EUC_FUNCTION_CALL_NAME:
          continue
        # found the function call response for the system long running request euc
        # function call
        request_euc_function_call_ids.add(function_call_response.id)
        auth_config = AuthConfig.model_validate(function_call_response.response)
        AuthHandler(auth_config=auth_config).parse_and_store_auth_response(
            state=invocation_context.session.state
        )
      break

    if not request_euc_function_call_ids:
      return

    for i in range(len(events) - 2, -1, -1):
      event = events[i]
      # looking for the system long running request euc function call
      function_calls = event.get_function_calls()
      if not function_calls:
        continue

      tools_to_resume = set()

      for function_call in function_calls:
        if function_call.id not in request_euc_function_call_ids:
          continue
        args = AuthToolArguments.model_validate(function_call.args)

        tools_to_resume.add(args.function_call_id)
      if not tools_to_resume:
        continue

      # found the the system long running request euc function call
      # looking for original function call that requests euc
      for j in range(i - 1, -1, -1):
        event = events[j]
        function_calls = event.get_function_calls()
        if not function_calls:
          continue
        for function_call in function_calls:
          function_response_event = None
          if function_call.id in tools_to_resume:
            function_response_event = await functions.handle_function_calls_async(
                invocation_context,
                event,
                {tool.name: tool for tool in agent.canonical_tools},
                # there could be parallel function calls that require auth
                # auth response would be a dict keyed by function call id
                tools_to_resume,
            )
          if function_response_event:
            yield function_response_event
          return
      return


request_processor = _AuthLlmRequestProcessor()



================================================
FILE: src/google/adk/auth/auth_schemes.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import List
from typing import Optional
from typing import Union

from fastapi.openapi.models import OAuthFlows
from fastapi.openapi.models import SecurityBase
from fastapi.openapi.models import SecurityScheme
from fastapi.openapi.models import SecuritySchemeType
from pydantic import Field


class OpenIdConnectWithConfig(SecurityBase):
  type_: SecuritySchemeType = Field(
      default=SecuritySchemeType.openIdConnect, alias="type"
  )
  authorization_endpoint: str
  token_endpoint: str
  userinfo_endpoint: Optional[str] = None
  revocation_endpoint: Optional[str] = None
  token_endpoint_auth_methods_supported: Optional[List[str]] = None
  grant_types_supported: Optional[List[str]] = None
  scopes: Optional[List[str]] = None


# AuthSchemes contains SecuritySchemes from OpenAPI 3.0 and an extra flattened OpenIdConnectWithConfig.
AuthScheme = Union[SecurityScheme, OpenIdConnectWithConfig]


class OAuthGrantType(str, Enum):
  """Represents the OAuth2 flow (or grant type)."""

  CLIENT_CREDENTIALS = "client_credentials"
  AUTHORIZATION_CODE = "authorization_code"
  IMPLICIT = "implicit"
  PASSWORD = "password"

  @staticmethod
  def from_flow(flow: OAuthFlows) -> "OAuthGrantType":
    """Converts an OAuthFlows object to a OAuthGrantType."""
    if flow.clientCredentials:
      return OAuthGrantType.CLIENT_CREDENTIALS
    if flow.authorizationCode:
      return OAuthGrantType.AUTHORIZATION_CODE
    if flow.implicit:
      return OAuthGrantType.IMPLICIT
    if flow.password:
      return OAuthGrantType.PASSWORD
    return None


# AuthSchemeType re-exports SecuritySchemeType from OpenAPI 3.0.
AuthSchemeType = SecuritySchemeType



================================================
FILE: src/google/adk/auth/auth_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pydantic import BaseModel

from .auth_credential import AuthCredential
from .auth_schemes import AuthScheme


class AuthConfig(BaseModel):
  """The auth config sent by tool asking client to collect auth credentials and

  adk and client will help to fill in the response
  """

  auth_scheme: AuthScheme
  """The auth scheme used to collect credentials"""
  raw_auth_credential: AuthCredential = None
  """The raw auth credential used to collect credentials. The raw auth
  credentials are used in some auth scheme that needs to exchange auth
  credentials. e.g. OAuth2 and OIDC. For other auth scheme, it could be None.
  """
  exchanged_auth_credential: AuthCredential = None
  """The exchanged auth credential used to collect credentials. adk and client
  will work together to fill it. For those auth scheme that doesn't need to
  exchange auth credentials, e.g. API key, service account etc. It's filled by
  client directly. For those auth scheme that need to exchange auth credentials,
  e.g. OAuth2 and OIDC, it's first filled by adk. If the raw credentials
  passed by tool only has client id and client credential, adk will help to
  generate the corresponding authorization uri and state and store the processed
  credential in this field. If the raw credentials passed by tool already has
  authorization uri, state, etc. then it's copied to this field. Client will use
  this field to guide the user through the OAuth2 flow and fill auth response in
  this field"""


class AuthToolArguments(BaseModel):
  """the arguments for the special long running function tool that is used to

  request end user credentials.
  """

  function_call_id: str
  auth_config: AuthConfig



================================================
FILE: src/google/adk/cli/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .cli_tools_click import main



================================================
FILE: src/google/adk/cli/__main__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .cli_tools_click import main

if __name__ == '__main__':
  main()



================================================
FILE: src/google/adk/cli/agent_graph.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Union

import graphviz

from ..agents import BaseAgent
from ..agents.llm_agent import LlmAgent
from ..tools.agent_tool import AgentTool
from ..tools.base_tool import BaseTool
from ..tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)

try:
  from ..tools.retrieval.base_retrieval_tool import BaseRetrievalTool
except ModuleNotFoundError:
  retrieval_tool_module_loaded = False
else:
  retrieval_tool_module_loaded = True


def build_graph(graph, agent: BaseAgent, highlight_pairs):
  dark_green = '#0F5223'
  light_green = '#69CB87'
  light_gray = '#cccccc'

  def get_node_name(tool_or_agent: Union[BaseAgent, BaseTool]):
    if isinstance(tool_or_agent, BaseAgent):
      return tool_or_agent.name
    elif isinstance(tool_or_agent, BaseTool):
      return tool_or_agent.name
    else:
      raise ValueError(f'Unsupported tool type: {tool_or_agent}')

  def get_node_caption(tool_or_agent: Union[BaseAgent, BaseTool]):

    if isinstance(tool_or_agent, BaseAgent):
      return '🤖 ' + tool_or_agent.name
    elif retrieval_tool_module_loaded and isinstance(
        tool_or_agent, BaseRetrievalTool
    ):
      return '🔎 ' + tool_or_agent.name
    elif isinstance(tool_or_agent, FunctionTool):
      return '🔧 ' + tool_or_agent.name
    elif isinstance(tool_or_agent, AgentTool):
      return '🤖 ' + tool_or_agent.name
    elif isinstance(tool_or_agent, BaseTool):
      return '🔧 ' + tool_or_agent.name
    else:
      logger.warning(
          'Unsupported tool, type: %s, obj: %s',
          type(tool_or_agent),
          tool_or_agent,
      )
      return f'❓ Unsupported tool type: {type(tool_or_agent)}'

  def get_node_shape(tool_or_agent: Union[BaseAgent, BaseTool]):
    if isinstance(tool_or_agent, BaseAgent):
      return 'ellipse'
    elif retrieval_tool_module_loaded and isinstance(
        tool_or_agent, BaseRetrievalTool
    ):
      return 'cylinder'
    elif isinstance(tool_or_agent, FunctionTool):
      return 'box'
    elif isinstance(tool_or_agent, BaseTool):
      return 'box'
    else:
      logger.warning(
          'Unsupported tool, type: %s, obj: %s',
          type(tool_or_agent),
          tool_or_agent,
      )
      return 'cylinder'

  def draw_node(tool_or_agent: Union[BaseAgent, BaseTool]):
    name = get_node_name(tool_or_agent)
    shape = get_node_shape(tool_or_agent)
    caption = get_node_caption(tool_or_agent)
    if highlight_pairs:
      for highlight_tuple in highlight_pairs:
        if name in highlight_tuple:
          graph.node(
              name,
              caption,
              style='filled,rounded',
              fillcolor=dark_green,
              color=dark_green,
              shape=shape,
              fontcolor=light_gray,
          )
          return
    # if not in highlight, draw non-highliht node
    graph.node(
        name,
        caption,
        shape=shape,
        style='rounded',
        color=light_gray,
        fontcolor=light_gray,
    )

  def draw_edge(from_name, to_name):
    if highlight_pairs:
      for highlight_from, highlight_to in highlight_pairs:
        if from_name == highlight_from and to_name == highlight_to:
          graph.edge(from_name, to_name, color=light_green)
          return
        elif from_name == highlight_to and to_name == highlight_from:
          graph.edge(from_name, to_name, color=light_green, dir='back')
          return
    # if no need to highlight, color gray
    graph.edge(from_name, to_name, arrowhead='none', color=light_gray)

  draw_node(agent)
  for sub_agent in agent.sub_agents:
    build_graph(graph, sub_agent, highlight_pairs)
    draw_edge(agent.name, sub_agent.name)
  if isinstance(agent, LlmAgent):
    for tool in agent.canonical_tools:
      draw_node(tool)
      draw_edge(agent.name, get_node_name(tool))


def get_agent_graph(root_agent, highlights_pairs, image=False):
  print('build graph')
  graph = graphviz.Digraph(graph_attr={'rankdir': 'LR', 'bgcolor': '#333537'})
  build_graph(graph, root_agent, highlights_pairs)
  if image:
    return graph.pipe(format='png')
  else:
    return graph



================================================
FILE: src/google/adk/cli/cli.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
import importlib
import os
import sys
from typing import Optional

import click
from google.genai import types
from pydantic import BaseModel

from ..agents.llm_agent import LlmAgent
from ..artifacts import BaseArtifactService
from ..artifacts import InMemoryArtifactService
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from .utils import envs


class InputFile(BaseModel):
  state: dict[str, object]
  queries: list[str]


async def run_input_file(
    app_name: str,
    user_id: str,
    root_agent: LlmAgent,
    artifact_service: BaseArtifactService,
    session_service: BaseSessionService,
    input_path: str,
) -> Session:
  runner = Runner(
      app_name=app_name,
      agent=root_agent,
      artifact_service=artifact_service,
      session_service=session_service,
  )
  with open(input_path, 'r', encoding='utf-8') as f:
    input_file = InputFile.model_validate_json(f.read())
  input_file.state['_time'] = datetime.now()

  session = session_service.create_session(
      app_name=app_name, user_id=user_id, state=input_file.state
  )
  for query in input_file.queries:
    click.echo(f'[user]: {query}')
    content = types.Content(role='user', parts=[types.Part(text=query)])
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
      if event.content and event.content.parts:
        if text := ''.join(part.text or '' for part in event.content.parts):
          click.echo(f'[{event.author}]: {text}')
  return session


async def run_interactively(
    root_agent: LlmAgent,
    artifact_service: BaseArtifactService,
    session: Session,
    session_service: BaseSessionService,
) -> None:
  runner = Runner(
      app_name=session.app_name,
      agent=root_agent,
      artifact_service=artifact_service,
      session_service=session_service,
  )
  while True:
    query = input('[user]: ')
    if not query or not query.strip():
      continue
    if query == 'exit':
      break
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=types.Content(role='user', parts=[types.Part(text=query)]),
    ):
      if event.content and event.content.parts:
        if text := ''.join(part.text or '' for part in event.content.parts):
          click.echo(f'[{event.author}]: {text}')


async def run_cli(
    *,
    agent_parent_dir: str,
    agent_folder_name: str,
    input_file: Optional[str] = None,
    saved_session_file: Optional[str] = None,
    save_session: bool,
) -> None:
  """Runs an interactive CLI for a certain agent.

  Args:
    agent_parent_dir: str, the absolute path of the parent folder of the agent
      folder.
    agent_folder_name: str, the name of the agent folder.
    input_file: Optional[str], the absolute path to the json file that contains
      the initial session state and user queries, exclusive with
      saved_session_file.
    saved_session_file: Optional[str], the absolute path to the json file that
      contains a previously saved session, exclusive with input_file.
    save_session: bool, whether to save the session on exit.
  """
  if agent_parent_dir not in sys.path:
    sys.path.append(agent_parent_dir)

  artifact_service = InMemoryArtifactService()
  session_service = InMemorySessionService()

  agent_module_path = os.path.join(agent_parent_dir, agent_folder_name)
  agent_module = importlib.import_module(agent_folder_name)
  user_id = 'test_user'
  session = session_service.create_session(
      app_name=agent_folder_name, user_id=user_id
  )
  root_agent = agent_module.agent.root_agent
  envs.load_dotenv_for_agent(agent_folder_name, agent_parent_dir)
  if input_file:
    session = await run_input_file(
        app_name=agent_folder_name,
        user_id=user_id,
        root_agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
        input_path=input_file,
    )
  elif saved_session_file:

    loaded_session = None
    with open(saved_session_file, 'r') as f:
      loaded_session = Session.model_validate_json(f.read())

    if loaded_session:
      for event in loaded_session.events:
        session_service.append_event(session, event)
        content = event.content
        if not content or not content.parts or not content.parts[0].text:
          continue
        if event.author == 'user':
          click.echo(f'[user]: {content.parts[0].text}')
        else:
          click.echo(f'[{event.author}]: {content.parts[0].text}')

    await run_interactively(
        root_agent,
        artifact_service,
        session,
        session_service,
    )
  else:
    click.echo(f'Running agent {root_agent.name}, type exit to exit.')
    await run_interactively(
        root_agent,
        artifact_service,
        session,
        session_service,
    )

  if save_session:
    session_id = input('Session ID to save: ')
    session_path = f'{agent_module_path}/{session_id}.session.json'

    # Fetch the session again to get all the details.
    session = session_service.get_session(
        app_name=session.app_name,
        user_id=session.user_id,
        session_id=session.id,
    )
    with open(session_path, 'w') as f:
      f.write(session.model_dump_json(indent=2, exclude_none=True))

    print('Session saved to', session_path)



================================================
FILE: src/google/adk/cli/cli_create.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from typing import Optional
from typing import Tuple

import click

_INIT_PY_TEMPLATE = """\
from . import agent
"""

_AGENT_PY_TEMPLATE = """\
from google.adk.agents import Agent

root_agent = Agent(
    model='{model_name}',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions to the best of your knowledge',
)
"""


_GOOGLE_API_MSG = """
Don't have API Key? Create one in AI Studio: https://aistudio.google.com/apikey
"""

_GOOGLE_CLOUD_SETUP_MSG = """
You need an existing Google Cloud account and project, check out this link for details:
https://google.github.io/adk-docs/get-started/quickstart/#gemini---google-cloud-vertex-ai
"""

_OTHER_MODEL_MSG = """
Please see below guide to configure other models:
https://google.github.io/adk-docs/agents/models
"""

_SUCCESS_MSG = """
Agent created in {agent_folder}:
- .env
- __init__.py
- agent.py
"""


def _get_gcp_project_from_gcloud() -> str:
  """Uses gcloud to get default project."""
  try:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return ""


def _get_gcp_region_from_gcloud() -> str:
  """Uses gcloud to get default region."""
  try:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "compute/region"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return ""


def _prompt_str(
    prompt_prefix: str,
    *,
    prior_msg: Optional[str] = None,
    default_value: Optional[str] = None,
) -> str:
  if prior_msg:
    click.secho(prior_msg, fg="green")
  while True:
    value: str = click.prompt(
        prompt_prefix, default=default_value or None, type=str
    )
    if value and value.strip():
      return value.strip()


def _prompt_for_google_cloud(
    google_cloud_project: Optional[str],
) -> str:
  """Prompts user for Google Cloud project ID."""
  google_cloud_project = (
      google_cloud_project
      or os.environ.get("GOOGLE_CLOUD_PROJECT", None)
      or _get_gcp_project_from_gcloud()
  )

  google_cloud_project = _prompt_str(
      "Enter Google Cloud project ID", default_value=google_cloud_project
  )

  return google_cloud_project


def _prompt_for_google_cloud_region(
    google_cloud_region: Optional[str],
) -> str:
  """Prompts user for Google Cloud region."""
  google_cloud_region = (
      google_cloud_region
      or os.environ.get("GOOGLE_CLOUD_LOCATION", None)
      or _get_gcp_region_from_gcloud()
  )

  google_cloud_region = _prompt_str(
      "Enter Google Cloud region",
      default_value=google_cloud_region or "us-central1",
  )
  return google_cloud_region


def _prompt_for_google_api_key(
    google_api_key: Optional[str],
) -> str:
  """Prompts user for Google API key."""
  google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY", None)

  google_api_key = _prompt_str(
      "Enter Google API key",
      prior_msg=_GOOGLE_API_MSG,
      default_value=google_api_key,
  )
  return google_api_key


def _generate_files(
    agent_folder: str,
    *,
    google_api_key: Optional[str] = None,
    google_cloud_project: Optional[str] = None,
    google_cloud_region: Optional[str] = None,
    model: Optional[str] = None,
):
  """Generates a folder name for the agent."""
  os.makedirs(agent_folder, exist_ok=True)

  dotenv_file_path = os.path.join(agent_folder, ".env")
  init_file_path = os.path.join(agent_folder, "__init__.py")
  agent_file_path = os.path.join(agent_folder, "agent.py")

  with open(dotenv_file_path, "w", encoding="utf-8") as f:
    lines = []
    if google_api_key:
      lines.append("GOOGLE_GENAI_USE_VERTEXAI=0")
    elif google_cloud_project and google_cloud_region:
      lines.append("GOOGLE_GENAI_USE_VERTEXAI=1")
    if google_api_key:
      lines.append(f"GOOGLE_API_KEY={google_api_key}")
    if google_cloud_project:
      lines.append(f"GOOGLE_CLOUD_PROJECT={google_cloud_project}")
    if google_cloud_region:
      lines.append(f"GOOGLE_CLOUD_LOCATION={google_cloud_region}")
    f.write("\n".join(lines))

  with open(init_file_path, "w", encoding="utf-8") as f:
    f.write(_INIT_PY_TEMPLATE)

  with open(agent_file_path, "w", encoding="utf-8") as f:
    f.write(_AGENT_PY_TEMPLATE.format(model_name=model))

  click.secho(
      _SUCCESS_MSG.format(agent_folder=agent_folder),
      fg="green",
  )


def _prompt_for_model() -> str:
  model_choice = click.prompt(
      """\
Choose a model for the root agent:
1. gemini-2.5-flash-preview-04-17
2. Other models (fill later)
Choose model""",
      type=click.Choice(["1", "2"]),
  )
  if model_choice == "1":
    return "gemini-2.5-flash-preview-04-17"
  else:
    click.secho(_OTHER_MODEL_MSG, fg="green")
    return "<FILL_IN_MODEL>"


def _prompt_to_choose_backend(
    google_api_key: Optional[str],
    google_cloud_project: Optional[str],
    google_cloud_region: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
  """Prompts user to choose backend.

  Returns:
    A tuple of (google_api_key, google_cloud_project, google_cloud_region).
  """
  backend_choice = click.prompt(
      "1. Google AI\n2. Vertex AI\nChoose a backend",
      type=click.Choice(["1", "2"]),
  )
  if backend_choice == "1":
    google_api_key = _prompt_for_google_api_key(google_api_key)
  elif backend_choice == "2":
    click.secho(_GOOGLE_CLOUD_SETUP_MSG, fg="green")
    google_cloud_project = _prompt_for_google_cloud(google_cloud_project)
    google_cloud_region = _prompt_for_google_cloud_region(google_cloud_region)
  return google_api_key, google_cloud_project, google_cloud_region


def run_cmd(
    agent_name: str,
    *,
    model: Optional[str],
    google_api_key: Optional[str],
    google_cloud_project: Optional[str],
    google_cloud_region: Optional[str],
):
  """Runs `adk create` command to create agent template.

  Args:
    agent_name: str, The name of the agent.
    google_api_key: Optional[str], The Google API key for using Google AI as
      backend.
    google_cloud_project: Optional[str], The Google Cloud project for using
      VertexAI as backend.
    google_cloud_region: Optional[str], The Google Cloud region for using
      VertexAI as backend.
  """
  agent_folder = os.path.join(os.getcwd(), agent_name)
  # check folder doesn't exist or it's empty. Otherwise, throw
  if os.path.exists(agent_folder) and os.listdir(agent_folder):
    # Prompt user whether to override existing files using click
    if not click.confirm(
        f"Non-empty folder already exist: '{agent_folder}'\n"
        "Override existing content?",
        default=False,
    ):
      raise click.Abort()

  if not model:
    model = _prompt_for_model()

  if not google_api_key and not (google_cloud_project and google_cloud_region):
    if model.startswith("gemini"):
      google_api_key, google_cloud_project, google_cloud_region = (
          _prompt_to_choose_backend(
              google_api_key, google_cloud_project, google_cloud_region
          )
      )

  _generate_files(
      agent_folder,
      google_api_key=google_api_key,
      google_cloud_project=google_cloud_project,
      google_cloud_region=google_cloud_region,
      model=model,
  )



================================================
FILE: src/google/adk/cli/cli_deploy.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
from typing import Optional

import click

_DOCKERFILE_TEMPLATE = """
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser

# Change ownership of /app to myuser
RUN chown -R myuser:myuser /app

# Switch to the non-root user
USER myuser

# Set up environment variables - Start
ENV PATH="/home/myuser/.local/bin:$PATH"

ENV GOOGLE_GENAI_USE_VERTEXAI=1
ENV GOOGLE_CLOUD_PROJECT={gcp_project_id}
ENV GOOGLE_CLOUD_LOCATION={gcp_region}

# Set up environment variables - End

# Install ADK - Start
RUN pip install google-adk
# Install ADK - End

# Copy agent - Start

COPY "agents/{app_name}/" "/app/agents/{app_name}/"
{install_agent_deps}

# Copy agent - End

EXPOSE {port}

CMD adk {command} --port={port} {session_db_option} {trace_to_cloud_option} "/app/agents"
"""


def _resolve_project(project_in_option: Optional[str]) -> str:
  if project_in_option:
    return project_in_option

  result = subprocess.run(
      ['gcloud', 'config', 'get-value', 'project'],
      check=True,
      capture_output=True,
      text=True,
  )
  project = result.stdout.strip()
  click.echo(f'Use default project: {project}')
  return project


def to_cloud_run(
    *,
    agent_folder: str,
    project: Optional[str],
    region: Optional[str],
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    with_ui: bool,
    verbosity: str,
    session_db_url: str,
):
  """Deploys an agent to Google Cloud Run.

  `agent_folder` should contain the following files:

  - __init__.py
  - agent.py
  - requirements.txt (optional, for additional dependencies)
  - ... (other required source files)

  The folder structure of temp_folder will be

  * dist/[google_adk wheel file]
  * agents/[app_name]/
    * agent source code from `agent_folder`

  Args:
    agent_folder: The folder (absolute path) containing the agent source code.
    project: Google Cloud project id.
    region: Google Cloud region.
    service_name: The service name in Cloud Run.
    app_name: The name of the app, by default, it's basename of `agent_folder`.
    temp_folder: The temp folder for the generated Cloud Run source files.
    port: The port of the ADK api server.
    trace_to_cloud: Whether to enable Cloud Trace.
    with_ui: Whether to deploy with UI.
    verbosity: The verbosity level of the CLI.
    session_db_url: The database URL to connect the session.
  """
  app_name = app_name or os.path.basename(agent_folder)

  click.echo(f'Start generating Cloud Run source files in {temp_folder}')

  # remove temp_folder if exists
  if os.path.exists(temp_folder):
    click.echo('Removing existing files')
    shutil.rmtree(temp_folder)

  try:
    # copy agent source code
    click.echo('Copying agent source code...')
    agent_src_path = os.path.join(temp_folder, 'agents', app_name)
    shutil.copytree(agent_folder, agent_src_path)
    requirements_txt_path = os.path.join(agent_src_path, 'requirements.txt')
    install_agent_deps = (
        f'RUN pip install -r "/app/agents/{app_name}/requirements.txt"'
        if os.path.exists(requirements_txt_path)
        else ''
    )
    click.echo('Copying agent source code complete.')

    # create Dockerfile
    click.echo('Creating Dockerfile...')
    dockerfile_content = _DOCKERFILE_TEMPLATE.format(
        gcp_project_id=project,
        gcp_region=region,
        app_name=app_name,
        port=port,
        command='web' if with_ui else 'api_server',
        install_agent_deps=install_agent_deps,
        session_db_option=f'--session_db_url={session_db_url}'
        if session_db_url
        else '',
        trace_to_cloud_option='--trace_to_cloud' if trace_to_cloud else '',
    )
    dockerfile_path = os.path.join(temp_folder, 'Dockerfile')
    os.makedirs(temp_folder, exist_ok=True)
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
      f.write(
          dockerfile_content,
      )
    click.echo(f'Creating Dockerfile complete: {dockerfile_path}')

    # Deploy to Cloud Run
    click.echo('Deploying to Cloud Run...')
    region_options = ['--region', region] if region else []
    project = _resolve_project(project)
    subprocess.run(
        [
            'gcloud',
            'run',
            'deploy',
            service_name,
            '--source',
            temp_folder,
            '--project',
            project,
            *region_options,
            '--port',
            str(port),
            '--verbosity',
            verbosity,
            '--labels',
            'created-by=adk',
        ],
        check=True,
    )
  finally:
    click.echo(f'Cleaning up the temp folder: {temp_folder}')
    shutil.rmtree(temp_folder)



================================================
FILE: src/google/adk/cli/cli_eval.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import importlib.util
import json
import logging
import os
import sys
import traceback
from typing import Any
from typing import Generator
from typing import Optional
import uuid

from pydantic import BaseModel

from ..agents import Agent

logger = logging.getLogger(__name__)


class EvalStatus(Enum):
  PASSED = 1
  FAILED = 2
  NOT_EVALUATED = 3


class EvalMetric(BaseModel):
  metric_name: str
  threshold: float


class EvalMetricResult(BaseModel):
  score: Optional[float]
  eval_status: EvalStatus


class EvalResult(BaseModel):
  eval_set_file: str
  eval_id: str
  final_eval_status: EvalStatus
  eval_metric_results: list[tuple[EvalMetric, EvalMetricResult]]
  session_id: str


MISSING_EVAL_DEPENDENCIES_MESSAGE = (
    "Eval module is not installed, please install via `pip install"
    " google-adk[eval]`."
)
TOOL_TRAJECTORY_SCORE_KEY = "tool_trajectory_avg_score"
RESPONSE_MATCH_SCORE_KEY = "response_match_score"
# This evaluation is not very stable.
# This is always optional unless explicitly specified.
RESPONSE_EVALUATION_SCORE_KEY = "response_evaluation_score"

EVAL_SESSION_ID_PREFIX = "___eval___session___"
DEFAULT_CRITERIA = {
    TOOL_TRAJECTORY_SCORE_KEY: 1.0,  # 1-point scale; 1.0 is perfect.
    RESPONSE_MATCH_SCORE_KEY: 0.8,
}


def _import_from_path(module_name, file_path):
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def _get_agent_module(agent_module_file_path: str):
  file_path = os.path.join(agent_module_file_path, "__init__.py")
  module_name = "agent"
  return _import_from_path(module_name, file_path)


def get_evaluation_criteria_or_default(
    eval_config_file_path: str,
) -> dict[str, float]:
  """Returns evaluation criteria from the config file, if present.

  Otherwise a default one is returned.
  """
  if eval_config_file_path:
    with open(eval_config_file_path, "r", encoding="utf-8") as f:
      config_data = json.load(f)

    if "criteria" in config_data and isinstance(config_data["criteria"], dict):
      evaluation_criteria = config_data["criteria"]
    else:
      raise ValueError(
          f"Invalid format for test_config.json at {eval_config_file_path}."
          " Expected a 'criteria' dictionary."
      )
  else:
    logger.info("No config file supplied. Using default criteria.")
    evaluation_criteria = DEFAULT_CRITERIA

  return evaluation_criteria


def get_root_agent(agent_module_file_path: str) -> Agent:
  """Returns root agent given the agent module."""
  agent_module = _get_agent_module(agent_module_file_path)
  root_agent = agent_module.agent.root_agent
  return root_agent


def try_get_reset_func(agent_module_file_path: str) -> Any:
  """Returns reset function for the agent, if present, given the agent module."""
  agent_module = _get_agent_module(agent_module_file_path)
  reset_func = getattr(agent_module.agent, "reset_data", None)
  return reset_func


def parse_and_get_evals_to_run(
    eval_set_file_path: tuple[str],
) -> dict[str, list[str]]:
  """Returns a dictionary of eval sets to evals that should be run."""
  eval_set_to_evals = {}
  for input_eval_set in eval_set_file_path:
    evals = []
    if ":" not in input_eval_set:
      eval_set_file = input_eval_set
    else:
      eval_set_file = input_eval_set.split(":")[0]
      evals = input_eval_set.split(":")[1].split(",")

    if eval_set_file not in eval_set_to_evals:
      eval_set_to_evals[eval_set_file] = []

    eval_set_to_evals[eval_set_file].extend(evals)

  return eval_set_to_evals


def run_evals(
    eval_set_to_evals: dict[str, list[str]],
    root_agent: Agent,
    reset_func: Optional[Any],
    eval_metrics: list[EvalMetric],
    session_service=None,
    artifact_service=None,
    print_detailed_results=False,
) -> Generator[EvalResult, None, None]:
  try:
    from ..evaluation.agent_evaluator import EvaluationGenerator
    from ..evaluation.response_evaluator import ResponseEvaluator
    from ..evaluation.trajectory_evaluator import TrajectoryEvaluator
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

  """Returns a summary of eval runs."""
  for eval_set_file, evals_to_run in eval_set_to_evals.items():
    with open(eval_set_file, "r", encoding="utf-8") as file:
      eval_items = json.load(file)  # Load JSON into a list

    assert eval_items, f"No eval data found in eval set file: {eval_set_file}"

    for eval_item in eval_items:
      eval_name = eval_item["name"]
      eval_data = eval_item["data"]
      initial_session = eval_item.get("initial_session", {})

      if evals_to_run and eval_name not in evals_to_run:
        continue

      try:
        print(f"Running Eval: {eval_set_file}:{eval_name}")
        session_id = f"{EVAL_SESSION_ID_PREFIX}{str(uuid.uuid4())}"

        scrape_result = EvaluationGenerator._process_query_with_root_agent(
            data=eval_data,
            root_agent=root_agent,
            reset_func=reset_func,
            initial_session=initial_session,
            session_id=session_id,
            session_service=session_service,
            artifact_service=artifact_service,
        )

        eval_metric_results = []
        for eval_metric in eval_metrics:
          eval_metric_result = None
          if eval_metric.metric_name == TOOL_TRAJECTORY_SCORE_KEY:
            score = TrajectoryEvaluator.evaluate(
                [scrape_result], print_detailed_results=print_detailed_results
            )
            eval_metric_result = _get_eval_metric_result(eval_metric, score)
          elif eval_metric.metric_name == RESPONSE_MATCH_SCORE_KEY:
            score = ResponseEvaluator.evaluate(
                [scrape_result],
                [RESPONSE_MATCH_SCORE_KEY],
                print_detailed_results=print_detailed_results,
            )
            eval_metric_result = _get_eval_metric_result(
                eval_metric, score["rouge_1/mean"].item()
            )
          elif eval_metric.metric_name == RESPONSE_EVALUATION_SCORE_KEY:
            score = ResponseEvaluator.evaluate(
                [scrape_result],
                [RESPONSE_EVALUATION_SCORE_KEY],
                print_detailed_results=print_detailed_results,
            )
            eval_metric_result = _get_eval_metric_result(
                eval_metric, score["coherence/mean"].item()
            )
          else:
            logger.warning("`%s` is not supported.", eval_metric.metric_name)
            eval_metric_results.append((
                eval_metric,
                EvalMetricResult(eval_status=EvalStatus.NOT_EVALUATED),
            ))

          eval_metric_results.append((
              eval_metric,
              eval_metric_result,
          ))
          _print_eval_metric_result(eval_metric, eval_metric_result)

        final_eval_status = EvalStatus.NOT_EVALUATED

        # Go over the all the eval statuses and mark the final eval status as
        # passed if all of them pass, otherwise mark the final eval status to
        # failed.
        for eval_metric_result in eval_metric_results:
          eval_status = eval_metric_result[1].eval_status
          if eval_status == EvalStatus.PASSED:
            final_eval_status = EvalStatus.PASSED
          elif eval_status == EvalStatus.NOT_EVALUATED:
            continue
          elif eval_status == EvalStatus.FAILED:
            final_eval_status = EvalStatus.FAILED
            break
          else:
            raise ValueError("Unknown eval status.")

        yield EvalResult(
            eval_set_file=eval_set_file,
            eval_id=eval_name,
            final_eval_status=final_eval_status,
            eval_metric_results=eval_metric_results,
            session_id=session_id,
        )

        if final_eval_status == EvalStatus.PASSED:
          result = "✅ Passed"
        else:
          result = "❌ Failed"

        print(f"Result: {result}\n")

      except Exception as e:
        print(f"Error: {e}")
        logger.info("Error: %s", str(traceback.format_exc()))


def _get_eval_metric_result(eval_metric, score):
  eval_status = (
      EvalStatus.PASSED if score >= eval_metric.threshold else EvalStatus.FAILED
  )
  return EvalMetricResult(score=score, eval_status=eval_status)


def _print_eval_metric_result(eval_metric, eval_metric_result):
  print(
      f"Metric: {eval_metric.metric_name}\tStatus:"
      f" {eval_metric_result.eval_status}\tScore:"
      f" {eval_metric_result.score}\tThreshold: {eval_metric.threshold}"
  )



================================================
FILE: src/google/adk/cli/cli_tools_click.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os
import tempfile
from typing import Optional

import click
from fastapi import FastAPI
import uvicorn

from . import cli_create
from . import cli_deploy
from .cli import run_cli
from .cli_eval import MISSING_EVAL_DEPENDENCIES_MESSAGE
from .fast_api import get_fast_api_app
from .utils import envs
from .utils import logs

logger = logging.getLogger(__name__)


@click.group(context_settings={"max_content_width": 240})
def main():
  """Agent Development Kit CLI tools."""
  pass


@main.group()
def deploy():
  """Deploys agent to hosted environments."""
  pass


@main.command("create")
@click.option(
    "--model",
    type=str,
    help="Optional. The model used for the root agent.",
)
@click.option(
    "--api_key",
    type=str,
    help=(
        "Optional. The API Key needed to access the model, e.g. Google AI API"
        " Key."
    ),
)
@click.option(
    "--project",
    type=str,
    help="Optional. The Google Cloud Project for using VertexAI as backend.",
)
@click.option(
    "--region",
    type=str,
    help="Optional. The Google Cloud Region for using VertexAI as backend.",
)
@click.argument("app_name", type=str, required=True)
def cli_create_cmd(
    app_name: str,
    model: Optional[str],
    api_key: Optional[str],
    project: Optional[str],
    region: Optional[str],
):
  """Creates a new app in the current folder with prepopulated agent template.

  APP_NAME: required, the folder of the agent source code.

  Example:

    adk create path/to/my_app
  """
  cli_create.run_cmd(
      app_name,
      model=model,
      google_api_key=api_key,
      google_cloud_project=project,
      google_cloud_region=region,
  )


def validate_exclusive(ctx, param, value):
  # Store the validated parameters in the context
  if not hasattr(ctx, "exclusive_opts"):
    ctx.exclusive_opts = {}

  # If this option has a value and we've already seen another exclusive option
  if value is not None and any(ctx.exclusive_opts.values()):
    exclusive_opt = next(key for key, val in ctx.exclusive_opts.items() if val)
    raise click.UsageError(
        f"Options '{param.name}' and '{exclusive_opt}' cannot be set together."
    )

  # Record this option's value
  ctx.exclusive_opts[param.name] = value is not None
  return value


@main.command("run")
@click.option(
    "--save_session",
    type=bool,
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to save the session to a json file on exit.",
)
@click.option(
    "--replay",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    help=(
        "The json file that contains the initial state of the session and user"
        " queries. A new session will be created using this state. And user"
        " queries are run againt the newly created session. Users cannot"
        " continue to interact with the agent."
    ),
    callback=validate_exclusive,
)
@click.option(
    "--resume",
    type=click.Path(
        exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    help=(
        "The json file that contains a previously saved session (by"
        "--save_session option). The previous session will be re-displayed. And"
        " user can continue to interact with the agent."
    ),
    callback=validate_exclusive,
)
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
def cli_run(
    agent: str,
    save_session: bool,
    replay: Optional[str],
    resume: Optional[str],
):
  """Runs an interactive CLI for a certain agent.

  AGENT: The path to the agent source code folder.

  Example:

    adk run path/to/my_agent
  """
  logs.log_to_tmp_folder()

  agent_parent_folder = os.path.dirname(agent)
  agent_folder_name = os.path.basename(agent)

  asyncio.run(
      run_cli(
          agent_parent_dir=agent_parent_folder,
          agent_folder_name=agent_folder_name,
          input_file=replay,
          saved_session_file=resume,
          save_session=save_session,
      )
  )


@main.command("eval")
@click.argument(
    "agent_module_file_path",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
@click.argument("eval_set_file_path", nargs=-1)
@click.option("--config_file_path", help="Optional. The path to config file.")
@click.option(
    "--print_detailed_results",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to print detailed results on console or not.",
)
def cli_eval(
    agent_module_file_path: str,
    eval_set_file_path: tuple[str],
    config_file_path: str,
    print_detailed_results: bool,
):
  """Evaluates an agent given the eval sets.

  AGENT_MODULE_FILE_PATH: The path to the __init__.py file that contains a
  module by the name "agent". "agent" module contains a root_agent.

  EVAL_SET_FILE_PATH: You can specify one or more eval set file paths.

  For each file, all evals will be run by default.

  If you want to run only specific evals from a eval set, first create a comma
  separated list of eval names and then add that as a suffix to the eval set
  file name, demarcated by a `:`.

  For example,

  sample_eval_set_file.json:eval_1,eval_2,eval_3

  This will only run eval_1, eval_2 and eval_3 from sample_eval_set_file.json.

  CONFIG_FILE_PATH: The path to config file.

  PRINT_DETAILED_RESULTS: Prints detailed results on the console.
  """
  envs.load_dotenv_for_agent(agent_module_file_path, ".")

  try:
    from .cli_eval import EvalMetric
    from .cli_eval import EvalResult
    from .cli_eval import EvalStatus
    from .cli_eval import get_evaluation_criteria_or_default
    from .cli_eval import get_root_agent
    from .cli_eval import parse_and_get_evals_to_run
    from .cli_eval import run_evals
    from .cli_eval import try_get_reset_func
  except ModuleNotFoundError:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE)

  evaluation_criteria = get_evaluation_criteria_or_default(config_file_path)
  eval_metrics = []
  for metric_name, threshold in evaluation_criteria.items():
    eval_metrics.append(
        EvalMetric(metric_name=metric_name, threshold=threshold)
    )

  print(f"Using evaluation creiteria: {evaluation_criteria}")

  root_agent = get_root_agent(agent_module_file_path)
  reset_func = try_get_reset_func(agent_module_file_path)

  eval_set_to_evals = parse_and_get_evals_to_run(eval_set_file_path)

  try:
    eval_results = list(
        run_evals(
            eval_set_to_evals,
            root_agent,
            reset_func,
            eval_metrics,
            print_detailed_results=print_detailed_results,
        )
    )
  except ModuleNotFoundError:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE)

  print("*********************************************************************")
  eval_run_summary = {}

  for eval_result in eval_results:
    eval_result: EvalResult

    if eval_result.eval_set_file not in eval_run_summary:
      eval_run_summary[eval_result.eval_set_file] = [0, 0]

    if eval_result.final_eval_status == EvalStatus.PASSED:
      eval_run_summary[eval_result.eval_set_file][0] += 1
    else:
      eval_run_summary[eval_result.eval_set_file][1] += 1
  print("Eval Run Summary")
  for eval_set_file, pass_fail_count in eval_run_summary.items():
    print(
        f"{eval_set_file}:\n  Tests passed: {pass_fail_count[0]}\n  Tests"
        f" failed: {pass_fail_count[1]}"
    )


@main.command("web")
@click.option(
    "--session_db_url",
    help=(
        """Optional. The database URL to store the session.

  - Use 'agentengine://<agent_engine_resource_id>' to connect to Agent Engine sessions.

  - Use 'sqlite://<path_to_sqlite_file>' to connect to a SQLite DB.

  - See https://docs.sqlalchemy.org/en/20/core/engines.html#backend-specific-urls for more details on supported DB URLs."""
    ),
)
@click.option(
    "--port",
    type=int,
    help="Optional. The port of the server",
    default=8000,
)
@click.option(
    "--allow_origins",
    help="Optional. Any additional origins to allow for CORS.",
    multiple=True,
)
@click.option(
    "--log_level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Optional. Set the logging level",
)
@click.option(
    "--log_to_tmp",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Whether to log to system temp folder instead of console."
        " This is useful for local debugging."
    ),
)
@click.option(
    "--trace_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable cloud trace for telemetry.",
)
@click.argument(
    "agents_dir",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    default=os.getcwd,
)
def cli_web(
    agents_dir: str,
    log_to_tmp: bool,
    session_db_url: str = "",
    log_level: str = "INFO",
    allow_origins: Optional[list[str]] = None,
    port: int = 8000,
    trace_to_cloud: bool = False,
):
  """Starts a FastAPI server with Web UI for agents.

  AGENTS_DIR: The directory of agents, where each sub-directory is a single
  agent, containing at least `__init__.py` and `agent.py` files.

  Example:

    adk web --session_db_url=[db_url] --port=[port] path/to/agents_dir
  """
  if log_to_tmp:
    logs.log_to_tmp_folder()
  else:
    logs.log_to_stderr()

  logging.getLogger().setLevel(log_level)

  @asynccontextmanager
  async def _lifespan(app: FastAPI):
    click.secho(
        f"""
+-----------------------------------------------------------------------------+
| ADK Web Server started                                                      |
|                                                                             |
| For local testing, access at http://localhost:{port}.{" "*(29 - len(str(port)))}|
+-----------------------------------------------------------------------------+
""",
        fg="green",
    )
    yield  # Startup is done, now app is running
    click.secho(
        """
+-----------------------------------------------------------------------------+
| ADK Web Server shutting down...                                             |
+-----------------------------------------------------------------------------+
""",
        fg="green",
    )

  app = get_fast_api_app(
      agent_dir=agents_dir,
      session_db_url=session_db_url,
      allow_origins=allow_origins,
      web=True,
      trace_to_cloud=trace_to_cloud,
      lifespan=_lifespan,
  )
  config = uvicorn.Config(
      app,
      host="0.0.0.0",
      port=port,
      reload=True,
  )

  server = uvicorn.Server(config)
  server.run()


@main.command("api_server")
@click.option(
    "--session_db_url",
    help=(
        """Optional. The database URL to store the session.

  - Use 'agentengine://<agent_engine_resource_id>' to connect to Agent Engine sessions.

  - Use 'sqlite://<path_to_sqlite_file>' to connect to a SQLite DB.

  - See https://docs.sqlalchemy.org/en/20/core/engines.html#backend-specific-urls for more details on supported DB URLs."""
    ),
)
@click.option(
    "--port",
    type=int,
    help="Optional. The port of the server",
    default=8000,
)
@click.option(
    "--allow_origins",
    help="Optional. Any additional origins to allow for CORS.",
    multiple=True,
)
@click.option(
    "--log_level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Optional. Set the logging level",
)
@click.option(
    "--log_to_tmp",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Whether to log to system temp folder instead of console."
        " This is useful for local debugging."
    ),
)
@click.option(
    "--trace_to_cloud",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable cloud trace for telemetry.",
)
# The directory of agents, where each sub-directory is a single agent.
# By default, it is the current working directory
@click.argument(
    "agents_dir",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    default=os.getcwd(),
)
def cli_api_server(
    agents_dir: str,
    log_to_tmp: bool,
    session_db_url: str = "",
    log_level: str = "INFO",
    allow_origins: Optional[list[str]] = None,
    port: int = 8000,
    trace_to_cloud: bool = False,
):
  """Starts a FastAPI server for agents.

  AGENTS_DIR: The directory of agents, where each sub-directory is a single
  agent, containing at least `__init__.py` and `agent.py` files.

  Example:

    adk api_server --session_db_url=[db_url] --port=[port] path/to/agents_dir
  """
  if log_to_tmp:
    logs.log_to_tmp_folder()
  else:
    logs.log_to_stderr()

  logging.getLogger().setLevel(log_level)

  config = uvicorn.Config(
      get_fast_api_app(
          agent_dir=agents_dir,
          session_db_url=session_db_url,
          allow_origins=allow_origins,
          web=False,
          trace_to_cloud=trace_to_cloud,
      ),
      host="0.0.0.0",
      port=port,
      reload=True,
  )
  server = uvicorn.Server(config)
  server.run()


@deploy.command("cloud_run")
@click.option(
    "--project",
    type=str,
    help=(
        "Required. Google Cloud project to deploy the agent. When absent,"
        " default project from gcloud config is used."
    ),
)
@click.option(
    "--region",
    type=str,
    help=(
        "Required. Google Cloud region to deploy the agent. When absent,"
        " gcloud run deploy will prompt later."
    ),
)
@click.option(
    "--service_name",
    type=str,
    default="adk-default-service-name",
    help=(
        "Optional. The service name to use in Cloud Run (default:"
        " 'adk-default-service-name')."
    ),
)
@click.option(
    "--app_name",
    type=str,
    default="",
    help=(
        "Optional. App name of the ADK API server (default: the folder name"
        " of the AGENT source code)."
    ),
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Optional. The port of the ADK API server (default: 8000).",
)
@click.option(
    "--trace_to_cloud",
    type=bool,
    is_flag=True,
    show_default=True,
    default=False,
    help="Optional. Whether to enable Cloud Trace for cloud run.",
)
@click.option(
    "--with_ui",
    type=bool,
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Optional. Deploy ADK Web UI if set. (default: deploy ADK API server"
        " only)"
    ),
)
@click.option(
    "--temp_folder",
    type=str,
    default=os.path.join(
        tempfile.gettempdir(),
        "cloud_run_deploy_src",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ),
    help=(
        "Optional. Temp folder for the generated Cloud Run source files"
        " (default: a timestamped folder in the system temp directory)."
    ),
)
@click.option(
    "--verbosity",
    type=click.Choice(
        ["debug", "info", "warning", "error", "critical"], case_sensitive=False
    ),
    default="WARNING",
    help="Optional. Override the default verbosity level.",
)
@click.option(
    "--session_db_url",
    help=(
        """Optional. The database URL to store the session.

  - Use 'agentengine://<agent_engine_resource_id>' to connect to Agent Engine sessions.

  - Use 'sqlite://<path_to_sqlite_file>' to connect to a SQLite DB.

  - See https://docs.sqlalchemy.org/en/20/core/engines.html#backend-specific-urls for more details on supported DB URLs."""
    ),
)
@click.argument(
    "agent",
    type=click.Path(
        exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
)
def cli_deploy_cloud_run(
    agent: str,
    project: Optional[str],
    region: Optional[str],
    service_name: str,
    app_name: str,
    temp_folder: str,
    port: int,
    trace_to_cloud: bool,
    with_ui: bool,
    verbosity: str,
    session_db_url: str,
):
  """Deploys an agent to Cloud Run.

  AGENT: The path to the agent source code folder.

  Example:

    adk deploy cloud_run --project=[project] --region=[region] path/to/my_agent
  """
  try:
    cli_deploy.to_cloud_run(
        agent_folder=agent,
        project=project,
        region=region,
        service_name=service_name,
        app_name=app_name,
        temp_folder=temp_folder,
        port=port,
        trace_to_cloud=trace_to_cloud,
        with_ui=with_ui,
        verbosity=verbosity,
        session_db_url=session_db_url,
    )
  except Exception as e:
    click.secho(f"Deploy failed: {e}", fg="red", err=True)



================================================
FILE: src/google/adk/cli/fast_api.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from contextlib import asynccontextmanager
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import sys
import traceback
import typing
from typing import Any
from typing import List
from typing import Literal
from typing import Optional

import click
from click import Tuple
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket
from fastapi.websockets import WebSocketDisconnect
from google.genai import types
import graphviz
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel
from pydantic import ValidationError
from starlette.types import Lifespan

from ..agents import RunConfig
from ..agents.live_request_queue import LiveRequest
from ..agents.live_request_queue import LiveRequestQueue
from ..agents.llm_agent import Agent
from ..agents.run_config import StreamingMode
from ..artifacts import InMemoryArtifactService
from ..events.event import Event
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..runners import Runner
from ..sessions.database_session_service import DatabaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from ..sessions.vertex_ai_session_service import VertexAiSessionService
from .cli_eval import EVAL_SESSION_ID_PREFIX
from .cli_eval import EvalMetric
from .cli_eval import EvalMetricResult
from .cli_eval import EvalStatus
from .utils import create_empty_state
from .utils import envs
from .utils import evals

logger = logging.getLogger(__name__)

_EVAL_SET_FILE_EXTENSION = ".evalset.json"


class ApiServerSpanExporter(export.SpanExporter):

  def __init__(self, trace_dict):
    self.trace_dict = trace_dict

  def export(
      self, spans: typing.Sequence[ReadableSpan]
  ) -> export.SpanExportResult:
    for span in spans:
      if (
          span.name == "call_llm"
          or span.name == "send_data"
          or span.name.startswith("tool_response")
      ):
        attributes = dict(span.attributes)
        attributes["trace_id"] = span.get_span_context().trace_id
        attributes["span_id"] = span.get_span_context().span_id
        if attributes.get("gcp.vertex.agent.event_id", None):
          self.trace_dict[attributes["gcp.vertex.agent.event_id"]] = attributes
    return export.SpanExportResult.SUCCESS

  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True


class AgentRunRequest(BaseModel):
  app_name: str
  user_id: str
  session_id: str
  new_message: types.Content
  streaming: bool = False


class AddSessionToEvalSetRequest(BaseModel):
  eval_id: str
  session_id: str
  user_id: str


class RunEvalRequest(BaseModel):
  eval_ids: list[str]  # if empty, then all evals in the eval set are run.
  eval_metrics: list[EvalMetric]


class RunEvalResult(BaseModel):
  eval_set_id: str
  eval_id: str
  final_eval_status: EvalStatus
  eval_metric_results: list[tuple[EvalMetric, EvalMetricResult]]
  session_id: str


def get_fast_api_app(
    *,
    agent_dir: str,
    session_db_url: str = "",
    allow_origins: Optional[list[str]] = None,
    web: bool,
    trace_to_cloud: bool = False,
    lifespan: Optional[Lifespan[FastAPI]] = None,
) -> FastAPI:
  # InMemory tracing dict.
  trace_dict: dict[str, Any] = {}

  # Set up tracing in the FastAPI server.
  provider = TracerProvider()
  provider.add_span_processor(
      export.SimpleSpanProcessor(ApiServerSpanExporter(trace_dict))
  )
  if trace_to_cloud:
    envs.load_dotenv_for_agent("", agent_dir)
    if project_id := os.environ.get("GOOGLE_CLOUD_PROJECT", None):
      processor = export.BatchSpanProcessor(
          CloudTraceSpanExporter(project_id=project_id)
      )
      provider.add_span_processor(processor)
    else:
      logging.warning(
          "GOOGLE_CLOUD_PROJECT environment variable is not set. Tracing will"
          " not be enabled."
      )

  trace.set_tracer_provider(provider)

  exit_stacks = []

  @asynccontextmanager
  async def internal_lifespan(app: FastAPI):
    if lifespan:
      async with lifespan(app) as lifespan_context:
        yield

        if exit_stacks:
          for stack in exit_stacks:
            await stack.aclose()
    else:
      yield

  # Run the FastAPI server.
  app = FastAPI(lifespan=internal_lifespan)

  if allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

  if agent_dir not in sys.path:
    sys.path.append(agent_dir)

  runner_dict = {}
  root_agent_dict = {}

  # Build the Artifact service
  artifact_service = InMemoryArtifactService()
  memory_service = InMemoryMemoryService()

  # Build the Session service
  agent_engine_id = ""
  if session_db_url:
    if session_db_url.startswith("agentengine://"):
      # Create vertex session service
      agent_engine_id = session_db_url.split("://")[1]
      if not agent_engine_id:
        raise click.ClickException("Agent engine id can not be empty.")
      envs.load_dotenv_for_agent("", agent_dir)
      session_service = VertexAiSessionService(
          os.environ["GOOGLE_CLOUD_PROJECT"],
          os.environ["GOOGLE_CLOUD_LOCATION"],
      )
    else:
      session_service = DatabaseSessionService(db_url=session_db_url)
  else:
    session_service = InMemorySessionService()

  @app.get("/list-apps")
  def list_apps() -> list[str]:
    base_path = Path.cwd() / agent_dir
    if not base_path.exists():
      raise HTTPException(status_code=404, detail="Path not found")
    if not base_path.is_dir():
      raise HTTPException(status_code=400, detail="Not a directory")
    agent_names = [
        x
        for x in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, x))
        and not x.startswith(".")
        and x != "__pycache__"
    ]
    agent_names.sort()
    return agent_names

  @app.get("/debug/trace/{event_id}")
  def get_trace_dict(event_id: str) -> Any:
    event_dict = trace_dict.get(event_id, None)
    if event_dict is None:
      raise HTTPException(status_code=404, detail="Trace not found")
    return event_dict

  @app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
      response_model_exclude_none=True,
  )
  def get_session(app_name: str, user_id: str, session_id: str) -> Session:
    # Connect to managed session if agent_engine_id is set.
    app_name = agent_engine_id if agent_engine_id else app_name
    session = session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if not session:
      raise HTTPException(status_code=404, detail="Session not found")
    return session

  @app.get(
      "/apps/{app_name}/users/{user_id}/sessions",
      response_model_exclude_none=True,
  )
  def list_sessions(app_name: str, user_id: str) -> list[Session]:
    # Connect to managed session if agent_engine_id is set.
    app_name = agent_engine_id if agent_engine_id else app_name
    return [
        session
        for session in session_service.list_sessions(
            app_name=app_name, user_id=user_id
        ).sessions
        # Remove sessions that were generated as a part of Eval.
        if not session.id.startswith(EVAL_SESSION_ID_PREFIX)
    ]

  @app.post(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
      response_model_exclude_none=True,
  )
  def create_session_with_id(
      app_name: str,
      user_id: str,
      session_id: str,
      state: Optional[dict[str, Any]] = None,
  ) -> Session:
    # Connect to managed session if agent_engine_id is set.
    app_name = agent_engine_id if agent_engine_id else app_name
    if (
        session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        is not None
    ):
      logger.warning("Session already exists: %s", session_id)
      raise HTTPException(
          status_code=400, detail=f"Session already exists: {session_id}"
      )

    logger.info("New session created: %s", session_id)
    return session_service.create_session(
        app_name=app_name, user_id=user_id, state=state, session_id=session_id
    )

  @app.post(
      "/apps/{app_name}/users/{user_id}/sessions",
      response_model_exclude_none=True,
  )
  def create_session(
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
  ) -> Session:
    # Connect to managed session if agent_engine_id is set.
    app_name = agent_engine_id if agent_engine_id else app_name

    logger.info("New session created")
    return session_service.create_session(
        app_name=app_name, user_id=user_id, state=state
    )

  def _get_eval_set_file_path(app_name, agent_dir, eval_set_id) -> str:
    return os.path.join(
        agent_dir,
        app_name,
        eval_set_id + _EVAL_SET_FILE_EXTENSION,
    )

  @app.post(
      "/apps/{app_name}/eval_sets/{eval_set_id}",
      response_model_exclude_none=True,
  )
  def create_eval_set(
      app_name: str,
      eval_set_id: str,
  ):
    """Creates an eval set, given the id."""
    pattern = r"^[a-zA-Z0-9_]+$"
    if not bool(re.fullmatch(pattern, eval_set_id)):
      raise HTTPException(
          status_code=400,
          detail=(
              f"Invalid eval set id. Eval set id should have the `{pattern}`"
              " format"
          ),
      )
    # Define the file path
    new_eval_set_path = _get_eval_set_file_path(
        app_name, agent_dir, eval_set_id
    )

    logger.info("Creating eval set file `%s`", new_eval_set_path)

    if not os.path.exists(new_eval_set_path):
      # Write the JSON string to the file
      logger.info("Eval set file doesn't exist, we will create a new one.")
      with open(new_eval_set_path, "w") as f:
        empty_content = json.dumps([], indent=2)
        f.write(empty_content)

  @app.get(
      "/apps/{app_name}/eval_sets",
      response_model_exclude_none=True,
  )
  def list_eval_sets(app_name: str) -> list[str]:
    """Lists all eval sets for the given app."""
    eval_set_file_path = os.path.join(agent_dir, app_name)
    eval_sets = []
    for file in os.listdir(eval_set_file_path):
      if file.endswith(_EVAL_SET_FILE_EXTENSION):
        eval_sets.append(
            os.path.basename(file).removesuffix(_EVAL_SET_FILE_EXTENSION)
        )

    return sorted(eval_sets)

  @app.post(
      "/apps/{app_name}/eval_sets/{eval_set_id}/add_session",
      response_model_exclude_none=True,
  )
  async def add_session_to_eval_set(
      app_name: str, eval_set_id: str, req: AddSessionToEvalSetRequest
  ):
    pattern = r"^[a-zA-Z0-9_]+$"
    if not bool(re.fullmatch(pattern, req.eval_id)):
      raise HTTPException(
          status_code=400,
          detail=f"Invalid eval id. Eval id should have the `{pattern}` format",
      )

    # Get the session
    session = session_service.get_session(
        app_name=app_name, user_id=req.user_id, session_id=req.session_id
    )
    assert session, "Session not found."
    # Load the eval set file data
    eval_set_file_path = _get_eval_set_file_path(
        app_name, agent_dir, eval_set_id
    )
    with open(eval_set_file_path, "r") as file:
      eval_set_data = json.load(file)  # Load JSON into a list

    if [x for x in eval_set_data if x["name"] == req.eval_id]:
      raise HTTPException(
          status_code=400,
          detail=(
              f"Eval id `{req.eval_id}` already exists in `{eval_set_id}`"
              " eval set."
          ),
      )

    # Convert the session data to evaluation format
    test_data = evals.convert_session_to_eval_format(session)

    # Populate the session with initial session state.
    initial_session_state = create_empty_state(
        await _get_root_agent_async(app_name)
    )

    eval_set_data.append({
        "name": req.eval_id,
        "data": test_data,
        "initial_session": {
            "state": initial_session_state,
            "app_name": app_name,
            "user_id": req.user_id,
        },
    })
    # Serialize the test data to JSON and write to the eval set file.
    with open(eval_set_file_path, "w") as f:
      f.write(json.dumps(eval_set_data, indent=2))

  @app.get(
      "/apps/{app_name}/eval_sets/{eval_set_id}/evals",
      response_model_exclude_none=True,
  )
  def list_evals_in_eval_set(
      app_name: str,
      eval_set_id: str,
  ) -> list[str]:
    """Lists all evals in an eval set."""
    # Load the eval set file data
    eval_set_file_path = _get_eval_set_file_path(
        app_name, agent_dir, eval_set_id
    )
    with open(eval_set_file_path, "r") as file:
      eval_set_data = json.load(file)  # Load JSON into a list

    return sorted([x["name"] for x in eval_set_data])

  @app.post(
      "/apps/{app_name}/eval_sets/{eval_set_id}/run_eval",
      response_model_exclude_none=True,
  )
  async def run_eval(
      app_name: str, eval_set_id: str, req: RunEvalRequest
  ) -> list[RunEvalResult]:
    from .cli_eval import run_evals

    """Runs an eval given the details in the eval request."""
    # Create a mapping from eval set file to all the evals that needed to be
    # run.
    eval_set_file_path = _get_eval_set_file_path(
        app_name, agent_dir, eval_set_id
    )
    eval_set_to_evals = {eval_set_file_path: req.eval_ids}

    if not req.eval_ids:
      logger.info(
          "Eval ids to run list is empty. We will all evals in the eval set."
      )
    root_agent = await _get_root_agent_async(app_name)
    eval_results = list(
        run_evals(
            eval_set_to_evals,
            root_agent,
            getattr(root_agent, "reset_data", None),
            req.eval_metrics,
            session_service=session_service,
            artifact_service=artifact_service,
        )
    )

    run_eval_results = []
    for eval_result in eval_results:
      run_eval_results.append(
          RunEvalResult(
              app_name=app_name,
              eval_set_id=eval_set_id,
              eval_id=eval_result.eval_id,
              final_eval_status=eval_result.final_eval_status,
              eval_metric_results=eval_result.eval_metric_results,
              session_id=eval_result.session_id,
          )
      )
    return run_eval_results

  @app.delete("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
  def delete_session(app_name: str, user_id: str, session_id: str):
    # Connect to managed session if agent_engine_id is set.
    app_name = agent_engine_id if agent_engine_id else app_name
    session_service.delete_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

  @app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
      response_model_exclude_none=True,
  )
  async def load_artifact(
      app_name: str,
      user_id: str,
      session_id: str,
      artifact_name: str,
      version: Optional[int] = Query(None),
  ) -> Optional[types.Part]:
    app_name = agent_engine_id if agent_engine_id else app_name
    artifact = await artifact_service.load_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=artifact_name,
        version=version,
    )
    if not artifact:
      raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact

  @app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version_id}",
      response_model_exclude_none=True,
  )
  async def load_artifact_version(
      app_name: str,
      user_id: str,
      session_id: str,
      artifact_name: str,
      version_id: int,
  ) -> Optional[types.Part]:
    app_name = agent_engine_id if agent_engine_id else app_name
    artifact = await artifact_service.load_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=artifact_name,
        version=version_id,
    )
    if not artifact:
      raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact

  @app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts",
      response_model_exclude_none=True,
  )
  async def list_artifact_names(
      app_name: str, user_id: str, session_id: str
  ) -> list[str]:
    app_name = agent_engine_id if agent_engine_id else app_name
    return await artifact_service.list_artifact_keys(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

  @app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions",
      response_model_exclude_none=True,
  )
  async def list_artifact_versions(
      app_name: str, user_id: str, session_id: str, artifact_name: str
  ) -> list[int]:
    app_name = agent_engine_id if agent_engine_id else app_name
    return await artifact_service.list_versions(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=artifact_name,
    )

  @app.delete(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
  )
  async def delete_artifact(
      app_name: str, user_id: str, session_id: str, artifact_name: str
  ):
    app_name = agent_engine_id if agent_engine_id else app_name
    await artifact_service.delete_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=artifact_name,
    )

  @app.post("/run", response_model_exclude_none=True)
  async def agent_run(req: AgentRunRequest) -> list[Event]:
    # Connect to managed session if agent_engine_id is set.
    app_id = agent_engine_id if agent_engine_id else req.app_name
    session = session_service.get_session(
        app_name=app_id, user_id=req.user_id, session_id=req.session_id
    )
    if not session:
      raise HTTPException(status_code=404, detail="Session not found")
    runner = await _get_runner_async(req.app_name)
    events = [
        event
        async for event in runner.run_async(
            user_id=req.user_id,
            session_id=req.session_id,
            new_message=req.new_message,
        )
    ]
    logger.info("Generated %s events in agent run: %s", len(events), events)
    return events

  @app.post("/run_sse")
  async def agent_run_sse(req: AgentRunRequest) -> StreamingResponse:
    # Connect to managed session if agent_engine_id is set.
    app_id = agent_engine_id if agent_engine_id else req.app_name
    # SSE endpoint
    session = session_service.get_session(
        app_name=app_id, user_id=req.user_id, session_id=req.session_id
    )
    if not session:
      raise HTTPException(status_code=404, detail="Session not found")

    # Convert the events to properly formatted SSE
    async def event_generator():
      try:
        stream_mode = StreamingMode.SSE if req.streaming else StreamingMode.NONE
        runner = await _get_runner_async(req.app_name)
        async for event in runner.run_async(
            user_id=req.user_id,
            session_id=req.session_id,
            new_message=req.new_message,
            run_config=RunConfig(streaming_mode=stream_mode),
        ):
          # Format as SSE data
          sse_event = event.model_dump_json(exclude_none=True, by_alias=True)
          logger.info("Generated event in agent run streaming: %s", sse_event)
          yield f"data: {sse_event}\n\n"
      except Exception as e:
        logger.exception("Error in event_generator: %s", e)
        # You might want to yield an error event here
        yield f'data: {{"error": "{str(e)}"}}\n\n'

    # Returns a streaming response with the proper media type for SSE
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

  @app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/events/{event_id}/graph",
      response_model_exclude_none=True,
  )
  async def get_event_graph(
      app_name: str, user_id: str, session_id: str, event_id: str
  ):
    # Connect to managed session if agent_engine_id is set.
    app_id = agent_engine_id if agent_engine_id else app_name
    session = session_service.get_session(
        app_name=app_id, user_id=user_id, session_id=session_id
    )
    session_events = session.events if session else []
    event = next((x for x in session_events if x.id == event_id), None)
    if not event:
      return {}

    from . import agent_graph

    function_calls = event.get_function_calls()
    function_responses = event.get_function_responses()
    root_agent = await _get_root_agent_async(app_name)
    dot_graph = None
    if function_calls:
      function_call_highlights = []
      for function_call in function_calls:
        from_name = event.author
        to_name = function_call.name
        function_call_highlights.append((from_name, to_name))
        dot_graph = agent_graph.get_agent_graph(
            root_agent, function_call_highlights
        )
    elif function_responses:
      function_responses_highlights = []
      for function_response in function_responses:
        from_name = function_response.name
        to_name = event.author
        function_responses_highlights.append((from_name, to_name))
        dot_graph = agent_graph.get_agent_graph(
            root_agent, function_responses_highlights
        )
    else:
      from_name = event.author
      to_name = ""
      dot_graph = agent_graph.get_agent_graph(
          root_agent, [(from_name, to_name)]
      )
    if dot_graph and isinstance(dot_graph, graphviz.Digraph):
      return {"dot_src": dot_graph.source}
    else:
      return {}

  @app.websocket("/run_live")
  async def agent_live_run(
      websocket: WebSocket,
      app_name: str,
      user_id: str,
      session_id: str,
      modalities: List[Literal["TEXT", "AUDIO"]] = Query(
          default=["TEXT", "AUDIO"]
      ),  # Only allows "TEXT" or "AUDIO"
  ) -> None:
    await websocket.accept()

    # Connect to managed session if agent_engine_id is set.
    app_id = agent_engine_id if agent_engine_id else app_name
    session = session_service.get_session(
        app_name=app_id, user_id=user_id, session_id=session_id
    )
    if not session:
      # Accept first so that the client is aware of connection establishment,
      # then close with a specific code.
      await websocket.close(code=1002, reason="Session not found")
      return

    live_request_queue = LiveRequestQueue()

    async def forward_events():
      runner = await _get_runner_async(app_name)
      async for event in runner.run_live(
          session=session, live_request_queue=live_request_queue
      ):
        await websocket.send_text(
            event.model_dump_json(exclude_none=True, by_alias=True)
        )

    async def process_messages():
      try:
        while True:
          data = await websocket.receive_text()
          # Validate and send the received message to the live queue.
          live_request_queue.send(LiveRequest.model_validate_json(data))
      except ValidationError as ve:
        logger.error("Validation error in process_messages: %s", ve)

    # Run both tasks concurrently and cancel all if one fails.
    tasks = [
        asyncio.create_task(forward_events()),
        asyncio.create_task(process_messages()),
    ]
    done, pending = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_EXCEPTION
    )
    try:
      # This will re-raise any exception from the completed tasks.
      for task in done:
        task.result()
    except WebSocketDisconnect:
      logger.info("Client disconnected during process_messages.")
    except Exception as e:
      logger.exception("Error during live websocket communication: %s", e)
      traceback.print_exc()
      WEBSOCKET_INTERNAL_ERROR_CODE = 1011
      WEBSOCKET_MAX_BYTES_FOR_REASON = 123
      await websocket.close(
          code=WEBSOCKET_INTERNAL_ERROR_CODE,
          reason=str(e)[:WEBSOCKET_MAX_BYTES_FOR_REASON],
      )
    finally:
      for task in pending:
        task.cancel()

  async def _get_root_agent_async(app_name: str) -> Agent:
    """Returns the root agent for the given app."""
    if app_name in root_agent_dict:
      return root_agent_dict[app_name]
    agent_module = importlib.import_module(app_name)
    if getattr(agent_module.agent, "root_agent"):
      root_agent = agent_module.agent.root_agent
    else:
      raise ValueError(f'Unable to find "root_agent" from {app_name}.')

    # Handle an awaitable root agent and await for the actual agent.
    if inspect.isawaitable(root_agent):
      try:
        agent, exit_stack = await root_agent
        exit_stacks.append(exit_stack)
        root_agent = agent
      except Exception as e:
        raise RuntimeError(f"error getting root agent, {e}") from e

    root_agent_dict[app_name] = root_agent
    return root_agent

  async def _get_runner_async(app_name: str) -> Runner:
    """Returns the runner for the given app."""
    envs.load_dotenv_for_agent(os.path.basename(app_name), agent_dir)
    if app_name in runner_dict:
      return runner_dict[app_name]
    root_agent = await _get_root_agent_async(app_name)
    runner = Runner(
        app_name=agent_engine_id if agent_engine_id else app_name,
        agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
    )
    runner_dict[app_name] = runner
    return runner

  if web:
    BASE_DIR = Path(__file__).parent.resolve()
    ANGULAR_DIST_PATH = BASE_DIR / "browser"

    @app.get("/")
    async def redirect_to_dev_ui():
      return RedirectResponse("/dev-ui")

    @app.get("/dev-ui")
    async def dev_ui():
      return FileResponse(BASE_DIR / "browser/index.html")

    app.mount(
        "/", StaticFiles(directory=ANGULAR_DIST_PATH, html=True), name="static"
    )
  return app


================================================
FILE: src/google/adk/cli/browser/assets/audio-processor.js
================================================
/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.targetSampleRate = 22000;  // Change to your desired rate
        this.originalSampleRate = sampleRate; // Browser's sample rate
        this.resampleRatio = this.originalSampleRate / this.targetSampleRate;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            let audioData = input[0]; // Get first channel's data
            
            if (this.resampleRatio !== 1) {
                audioData = this.resample(audioData);
            }

            this.port.postMessage(audioData);
        }
        return true; // Keep processor alive
    }

    resample(audioData) {
        const newLength = Math.round(audioData.length / this.resampleRatio);
        const resampled = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const srcIndex = Math.floor(i * this.resampleRatio);
            resampled[i] = audioData[srcIndex]; // Nearest neighbor resampling
        }
        return resampled;
    }
}

registerProcessor('audio-processor', AudioProcessor);



================================================
FILE: src/google/adk/cli/browser/assets/config/runtime-config.json
================================================
{
  "backendUrl": ""
}


================================================
FILE: src/google/adk/cli/utils/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any
from typing import Optional

from ...agents.base_agent import BaseAgent
from ...agents.llm_agent import LlmAgent

__all__ = [
    'create_empty_state',
]


def _create_empty_state(agent: BaseAgent, all_state: dict[str, Any]):
  for sub_agent in agent.sub_agents:
    _create_empty_state(sub_agent, all_state)

  if (
      isinstance(agent, LlmAgent)
      and agent.instruction
      and isinstance(agent.instruction, str)
  ):
    for key in re.findall(r'{([\w]+)}', agent.instruction):
      all_state[key] = ''


def create_empty_state(
    agent: BaseAgent, initialized_states: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
  """Creates empty str for non-initialized states."""
  non_initialized_states = {}
  _create_empty_state(agent, non_initialized_states)
  for key in initialized_states or {}:
    if key in non_initialized_states:
      del non_initialized_states[key]
  return non_initialized_states



================================================
FILE: src/google/adk/cli/utils/envs.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__file__)


def _walk_to_root_until_found(folder, filename) -> str:
  checkpath = os.path.join(folder, filename)
  if os.path.exists(checkpath) and os.path.isfile(checkpath):
    return checkpath

  parent_folder = os.path.dirname(folder)
  if parent_folder == folder:  # reached the root
    return ''

  return _walk_to_root_until_found(parent_folder, filename)


def load_dotenv_for_agent(
    agent_name: str, agent_parent_folder: str, filename: str = '.env'
):
  """Lods the .env file for the agent module."""

  # Gets the folder of agent_module as starting_folder
  starting_folder = os.path.abspath(
      os.path.join(agent_parent_folder, agent_name)
  )
  dotenv_file_path = _walk_to_root_until_found(starting_folder, filename)
  if dotenv_file_path:
    load_dotenv(dotenv_file_path, override=True, verbose=True)
    logger.info(
        'Loaded %s file for %s at %s',
        filename,
        agent_name,
        dotenv_file_path,
    )
  else:
    logger.info('No %s file found for %s', filename, agent_name)



================================================
FILE: src/google/adk/cli/utils/evals.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from ...sessions.session import Session


def convert_session_to_eval_format(session: Session) -> list[dict[str, Any]]:
  """Converts a session data into eval format.

  Args:
      session: The session that should be converted.

  Returns:
      list: A single evaluation dataset in the required format.
  """
  eval_case = []
  events = session.events if session and session.events else []

  for event in events:
    if event.author == 'user':
      if not event.content or not event.content.parts:
        continue

      # Extract user query
      content = event.content
      parts = content.parts

      query = parts[0].text or ''

      # Find the corresponding tool usage or response for the query
      expected_tool_use = []
      intermediate_agent_responses = []

      # Check subsequent events to extract tool uses or responses for this turn.
      for subsequent_event in events[events.index(event) + 1 :]:
        event_author = subsequent_event.author or 'agent'
        if event_author == 'user':
          # We found an event where the author was the user. This means that a
          # new turn has started. So close this turn here.
          break

        if not subsequent_event.content or not subsequent_event.content.parts:
          continue

        for subsequent_part in subsequent_event.content.parts:
          # Some events have both function call and reference

          if subsequent_part.function_call:
            tool_name = subsequent_part.function_call.name or ''
            tool_input = subsequent_part.function_call.args or {}
            expected_tool_use.append({
                'tool_name': tool_name,
                'tool_input': tool_input,
            })
          elif subsequent_part.text:
            # Also keep track of all the natural language responses that
            # agent (or sub agents) generated.
            intermediate_agent_responses.append(
                {'author': event_author, 'text': subsequent_part.text}
            )

      # If we are here then either we are done reading all the events or we
      # encountered an event that had content authored by the end-user.
      # This, basically means an end of turn.
      # We assume that the last natural language intermediate response is the
      # final response from the agent/model. We treat that as a reference.
      eval_case.append({
          'query': query,
          'expected_tool_use': expected_tool_use,
          'expected_intermediate_agent_responses': intermediate_agent_responses[
              :-1
          ],
          'reference': (
              intermediate_agent_responses[-1]['text']
              if intermediate_agent_responses
              else ''
          ),
      })

  return eval_case



================================================
FILE: src/google/adk/cli/utils/logs.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import tempfile
import time

LOGGING_FORMAT = (
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def log_to_stderr(level=logging.INFO):
  logging.basicConfig(
      level=level,
      format=LOGGING_FORMAT,
  )


def log_to_tmp_folder(
    level=logging.INFO,
    *,
    sub_folder: str = 'agents_log',
    log_file_prefix: str = 'agent',
    log_file_timestamp: str = time.strftime('%Y%m%d_%H%M%S'),
):
  """Logs to system temp folder, instead of logging to stderr.

  Args
    sub_folder: str = 'agents_log',
    log_file_prefix: str = 'agent',
    log_file_timestamp: str = time.strftime('%Y%m%d_%H%M%S'),

  Returns
    the log file path.
  """
  log_dir = os.path.join(tempfile.gettempdir(), sub_folder)
  log_filename = f'{log_file_prefix}.{log_file_timestamp}.log'
  log_filepath = os.path.join(log_dir, log_filename)

  os.makedirs(log_dir, exist_ok=True)

  file_handler = logging.FileHandler(log_filepath, mode='w')
  file_handler.setLevel(level)
  file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))

  root_logger = logging.getLogger()
  root_logger.setLevel(level)
  root_logger.handlers = []  # Clear handles to disable logging to stderr
  root_logger.addHandler(file_handler)

  print(f'Log setup complete: {log_filepath}')

  latest_log_link = os.path.join(log_dir, f'{log_file_prefix}.latest.log')
  if os.path.islink(latest_log_link):
    os.unlink(latest_log_link)
  os.symlink(log_filepath, latest_log_link)

  print(f'To access latest log: tail -F {latest_log_link}')
  return log_filepath



================================================
FILE: src/google/adk/code_executors/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from .base_code_executor import BaseCodeExecutor
from .code_executor_context import CodeExecutorContext
from .unsafe_local_code_executor import UnsafeLocalCodeExecutor

logger = logging.getLogger(__name__)

__all__ = [
    'BaseCodeExecutor',
    'CodeExecutorContext',
    'UnsafeLocalCodeExecutor',
]

try:
  from .vertex_ai_code_executor import VertexAiCodeExecutor

  __all__.append('VertexAiCodeExecutor')
except ImportError:
  logger.debug(
      'The Vertex sdk is not installed. If you want to use the Vertex Code'
      ' Interpreter with agents, please install it. If not, you can ignore this'
      ' warning.'
  )

try:
  from .container_code_executor import ContainerCodeExecutor

  __all__.append('ContainerCodeExecutor')
except ImportError:
  logger.debug(
      'The docker sdk is not installed. If you want to use the Container Code'
      ' Executor with agents, please install it. If not, you can ignore this'
      ' warning.'
  )



================================================
FILE: src/google/adk/code_executors/base_code_executor.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import List

from pydantic import BaseModel

from ..agents.invocation_context import InvocationContext
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult


class BaseCodeExecutor(BaseModel):
  """Abstract base class for all code executors.

  The code executor allows the agent to execute code blocks from model responses
  and incorporate the execution results into the final response.

  Attributes:
    optimize_data_file: If true, extract and process data files from the model
      request and attach them to the code executor. Supported data file
      MimeTypes are [text/csv]. Default to False.
    stateful: Whether the code executor is stateful. Default to False.
    error_retry_attempts: The number of attempts to retry on consecutive code
      execution errors. Default to 2.
    code_block_delimiters: The list of the enclosing delimiters to identify the
      code blocks.
    execution_result_delimiters: The delimiters to format the code execution
      result.
  """

  optimize_data_file: bool = False
  """
  If true, extract and process data files from the model request
  and attach them to the code executor.
  Supported data file MimeTypes are [text/csv].

  Default to False.
  """

  stateful: bool = False
  """
  Whether the code executor is stateful. Default to False.
  """

  error_retry_attempts: int = 2
  """
  The number of attempts to retry on consecutive code execution errors. Default to 2.
  """

  code_block_delimiters: List[tuple[str, str]] = [
      ('```tool_code\n', '\n```'),
      ('```python\n', '\n```'),
  ]
  """
  The list of the enclosing delimiters to identify the code blocks.
  For example, the delimiter ('```python\n', '\n```') can be
  used to identify code blocks with the following format:

  ```python
  print("hello")
  ```
  """

  execution_result_delimiters: tuple[str, str] = ('```tool_output\n', '\n```')
  """
  The delimiters to format the code execution result.
  """

  @abc.abstractmethod
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    """Executes code and return the code execution result.

    Args:
      invocation_context: The invocation context of the code execution.
      code_execution_input: The code execution input.

    Returns:
      The code execution result.
    """
    pass



================================================
FILE: src/google/adk/code_executors/code_execution_utils.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for code execution."""

import base64
import binascii
import copy
import dataclasses
import re
from typing import List, Optional

from google.genai import types


@dataclasses.dataclass(frozen=True)
class File:
  """A structure that contains a file name and its content."""

  name: str
  """
  The name of the file with file extension (e.g., "file.csv").
  """

  content: str
  """
  The base64-encoded bytes of the file content.
  """

  mime_type: str = 'text/plain'
  """
  The mime type of the file (e.g., "image/png").
  """


@dataclasses.dataclass
class CodeExecutionInput:
  """A structure that contains the input of code execution."""

  code: str
  """
  The code to execute.
  """

  input_files: list[File] = dataclasses.field(default_factory=list)
  """
  The input files available to the code.
  """

  execution_id: Optional[str] = None
  """
  The execution ID for the stateful code execution.
  """


@dataclasses.dataclass
class CodeExecutionResult:
  """A structure that contains the result of code execution."""

  stdout: str = ''
  """
  The standard output of the code execution.
  """

  stderr: str = ''
  """
  The standard error of the code execution.
  """

  output_files: list[File] = dataclasses.field(default_factory=list)
  """
  The output files from the code execution.
  """


class CodeExecutionUtils:
  """Utility functions for code execution."""

  @staticmethod
  def get_encoded_file_content(data: bytes) -> bytes:
    """Gets the file content as a base64-encoded bytes.

    Args:
      data: The file content bytes.

    Returns:
      The file content as a base64-encoded bytes.
    """

    def _is_base64_encoded(data: bytes) -> bool:
      try:
        return base64.b64encode(base64.b64decode(data)) == data
      except binascii.Error:
        return False

    return data if _is_base64_encoded(data) else base64.b64encode(data)

  @staticmethod
  def extract_code_and_truncate_content(
      content: types.Content,
      code_block_delimiters: List[tuple[str, str]],
  ) -> Optional[str]:
    """Extracts the first code block from the content and truncate everything after it.

    Args:
      content: The mutable content to extract the code from.
      code_block_delimiters: The list of the enclosing delimiters to identify
        the code blocks.

    Returns:
      The first code block if found, otherwise None.
    """
    if not content or not content.parts:
      return

    # Extract the code from the executable code parts if there're no associated
    # code execution result parts.
    for idx, part in enumerate(content.parts):
      if part.executable_code and (
          idx == len(content.parts) - 1
          or not content.parts[idx + 1].code_execution_result
      ):
        content.parts = content.parts[: idx + 1]
        return part.executable_code.code

    # Extract the code from the text parts.
    text_parts = [p for p in content.parts if p.text]
    if not text_parts:
      return

    first_text_part = copy.deepcopy(text_parts[0])
    response_text = '\n'.join([p.text for p in text_parts])

    # Find the first code block.
    leading_delimiter_pattern = '|'.join(d[0] for d in code_block_delimiters)
    trailing_delimiter_pattern = '|'.join(d[1] for d in code_block_delimiters)
    pattern = re.compile(
        (
            rf'(?P<prefix>.*?)({leading_delimiter_pattern})(?P<code>.*?)({trailing_delimiter_pattern})(?P<suffix>.*?)$'
        ).encode(),
        re.DOTALL,
    )
    pattern_match = pattern.search(response_text.encode())
    if pattern_match is None:
      return

    code_str = pattern_match.group('code').decode()
    if not code_str:
      return

    content.parts = []
    if pattern_match.group('prefix'):
      first_text_part.text = pattern_match.group('prefix').decode()
      content.parts.append(first_text_part)
    content.parts.append(
        CodeExecutionUtils.build_executable_code_part(code_str)
    )
    return pattern_match.group('code').decode()

  @staticmethod
  def build_executable_code_part(code: str) -> types.Part:
    """Builds an executable code part with code string.

    Args:
      code: The code string.

    Returns:
      The constructed executable code part.
    """
    return types.Part.from_executable_code(
        code=code,
        language='PYTHON',
    )

  @staticmethod
  def build_code_execution_result_part(
      code_execution_result: CodeExecutionResult,
  ) -> types.Part:
    """Builds the code execution result part from the code execution result.

    Args:
      code_execution_result: The code execution result.

    Returns:
      The constructed code execution result part.
    """
    if code_execution_result.stderr:
      return types.Part.from_code_execution_result(
          outcome='OUTCOME_FAILED',
          output=code_execution_result.stderr,
      )
    final_result = []
    if code_execution_result.stdout or not code_execution_result.output_files:
      final_result.append(
          'Code execution result:\n' + '%s\n' % code_execution_result.stdout
      )
    if code_execution_result.output_files:
      final_result.append(
          'Saved artifacts:\n'
          + ','.join(
              ['`%s`' % f.name for f in code_execution_result.output_files]
          )
      )
    return types.Part.from_code_execution_result(
        outcome='OUTCOME_OK',
        output='\n\n'.join(final_result),
    )

  @staticmethod
  def convert_code_execution_parts(
      content: types.Content,
      code_block_delimiter: tuple[str, str],
      execution_result_delimiters: tuple[str, str],
  ):
    """Converts the code execution parts to text parts in a Content.

    Args:
      content: The mutable content to convert the code execution parts to text
        parts.
      code_block_delimiter: The delimiter to format the code block.
      execution_result_delimiters: The delimiter to format the code execution
        result.
    """
    if not content.parts:
      return

    # Handle the conversion of trailing executable code parts.
    if content.parts[-1].executable_code:
      content.parts[-1] = types.Part(
          text=(
              code_block_delimiter[0]
              + content.parts[-1].executable_code.code
              + code_block_delimiter[1]
          )
      )
    # Handle the conversion of trailing code execution result parts.
    # Skip if the Content has multiple parts, which means the Content is
    # likely generated by the model.
    elif len(content.parts) == 1 and content.parts[-1].code_execution_result:
      content.parts[-1] = types.Part(
          text=execution_result_delimiters[0]
          + content.parts[-1].code_execution_result.output
          + execution_result_delimiters[1]
      )
      content.role = 'user'



================================================
FILE: src/google/adk/code_executors/code_executor_context.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The persistent context used to configure the code executor."""

import copy
import dataclasses
import datetime
from typing import Any
from typing import Optional

from ..sessions.state import State
from .code_execution_utils import File

_CONTEXT_KEY = '_code_execution_context'
_SESSION_ID_KEY = 'execution_session_id'
_PROCESSED_FILE_NAMES_KEY = 'processed_input_files'
_INPUT_FILE_KEY = '_code_executor_input_files'
_ERROR_COUNT_KEY = '_code_executor_error_counts'

_CODE_EXECUTION_RESULTS_KEY = '_code_execution_results'


class CodeExecutorContext:
  """The persistent context used to configure the code executor."""

  _context: dict[str, Any]

  def __init__(self, session_state: State):
    """Initializes the code executor context.

    Args:
      session_state: The session state to get the code executor context from.
    """
    self._context = self._get_code_executor_context(session_state)
    self._session_state = session_state

  def get_state_delta(self) -> dict[str, Any]:
    """Gets the state delta to update in the persistent session state.

    Returns:
      The state delta to update in the persistent session state.
    """
    context_to_update = copy.deepcopy(self._context)
    return {_CONTEXT_KEY: context_to_update}

  def get_execution_id(self) -> Optional[str]:
    """Gets the session ID for the code executor.

    Returns:
      The session ID for the code executor context.
    """
    if _SESSION_ID_KEY not in self._context:
      return None
    return self._context[_SESSION_ID_KEY]

  def set_execution_id(self, session_id: str):
    """Sets the session ID for the code executor.

    Args:
      session_id: The session ID for the code executor.
    """
    self._context[_SESSION_ID_KEY] = session_id

  def get_processed_file_names(self) -> list[str]:
    """Gets the processed file names from the session state.

    Returns:
      A list of processed file names in the code executor context.
    """
    if _PROCESSED_FILE_NAMES_KEY not in self._context:
      return []
    return self._context[_PROCESSED_FILE_NAMES_KEY]

  def add_processed_file_names(self, file_names: [str]):
    """Adds the processed file name to the session state.

    Args:
      file_names: The processed file names to add to the session state.
    """
    if _PROCESSED_FILE_NAMES_KEY not in self._context:
      self._context[_PROCESSED_FILE_NAMES_KEY] = []
    self._context[_PROCESSED_FILE_NAMES_KEY].extend(file_names)

  def get_input_files(self) -> list[File]:
    """Gets the code executor input file names from the session state.

    Returns:
      A list of input files in the code executor context.
    """
    if _INPUT_FILE_KEY not in self._session_state:
      return []
    return [File(**file) for file in self._session_state[_INPUT_FILE_KEY]]

  def add_input_files(
      self,
      input_files: list[File],
  ):
    """Adds the input files to the code executor context.

    Args:
      input_files: The input files to add to the code executor context.
    """
    if _INPUT_FILE_KEY not in self._session_state:
      self._session_state[_INPUT_FILE_KEY] = []
    for input_file in input_files:
      self._session_state[_INPUT_FILE_KEY].append(
          dataclasses.asdict(input_file)
      )

  def clear_input_files(self):
    """Removes the input files and processed file names to the code executor context."""
    if _INPUT_FILE_KEY in self._session_state:
      self._session_state[_INPUT_FILE_KEY] = []
    if _PROCESSED_FILE_NAMES_KEY in self._context:
      self._context[_PROCESSED_FILE_NAMES_KEY] = []

  def get_error_count(self, invocation_id: str) -> int:
    """Gets the error count from the session state.

    Args:
      invocation_id: The invocation ID to get the error count for.

    Returns:
      The error count for the given invocation ID.
    """
    if _ERROR_COUNT_KEY not in self._session_state:
      return 0
    return self._session_state[_ERROR_COUNT_KEY].get(invocation_id, 0)

  def increment_error_count(self, invocation_id: str):
    """Increments the error count from the session state.

    Args:
      invocation_id: The invocation ID to increment the error count for.
    """
    if _ERROR_COUNT_KEY not in self._session_state:
      self._session_state[_ERROR_COUNT_KEY] = {}
    self._session_state[_ERROR_COUNT_KEY][invocation_id] = (
        self.get_error_count(invocation_id) + 1
    )

  def reset_error_count(self, invocation_id: str):
    """Resets the error count from the session state.

    Args:
      invocation_id: The invocation ID to reset the error count for.
    """
    if _ERROR_COUNT_KEY not in self._session_state:
      return
    if invocation_id in self._session_state[_ERROR_COUNT_KEY]:
      del self._session_state[_ERROR_COUNT_KEY][invocation_id]

  def update_code_execution_result(
      self,
      invocation_id: str,
      code: str,
      result_stdout: str,
      result_stderr: str,
  ):
    """Updates the code execution result.

    Args:
      invocation_id: The invocation ID to update the code execution result for.
      code: The code to execute.
      result_stdout: The standard output of the code execution.
      result_stderr: The standard error of the code execution.
    """
    if _CODE_EXECUTION_RESULTS_KEY not in self._session_state:
      self._session_state[_CODE_EXECUTION_RESULTS_KEY] = {}
    if invocation_id not in self._session_state[_CODE_EXECUTION_RESULTS_KEY]:
      self._session_state[_CODE_EXECUTION_RESULTS_KEY][invocation_id] = []
    self._session_state[_CODE_EXECUTION_RESULTS_KEY][invocation_id].append({
        'code': code,
        'result_stdout': result_stdout,
        'result_stderr': result_stderr,
        'timestamp': int(datetime.datetime.now().timestamp()),
    })

  def _get_code_executor_context(self, session_state: State) -> dict[str, Any]:
    """Gets the code executor context from the session state.

    Args:
      session_state: The session state to get the code executor context from.

    Returns:
      A dict of code executor context.
    """
    if _CONTEXT_KEY not in session_state:
      session_state[_CONTEXT_KEY] = {}
    return session_state[_CONTEXT_KEY]



================================================
FILE: src/google/adk/code_executors/container_code_executor.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
from typing import Optional

import docker
from docker.client import DockerClient
from docker.models.containers import Container
from pydantic import Field
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult


DEFAULT_IMAGE_TAG = 'adk-code-executor:latest'


class ContainerCodeExecutor(BaseCodeExecutor):
  """A code executor that uses a custom container to execute code.

  Attributes:
    base_url: Optional. The base url of the user hosted Docker client.
    image: The tag of the predefined image or custom image to run on the
      container. Either docker_path or image must be set.
    docker_path: The path to the directory containing the Dockerfile. If set,
      build the image from the dockerfile path instead of using the predefined
      image. Either docker_path or image must be set.
  """

  base_url: Optional[str] = None
  """
  Optional. The base url of the user hosted Docker client.
  """

  image: str = None
  """
  The tag of the predefined image or custom image to run on the container.
  Either docker_path or image must be set.
  """

  docker_path: str = None
  """
  The path to the directory containing the Dockerfile.
  If set, build the image from the dockerfile path instead of using the
  predefined image. Either docker_path or image must be set.
  """

  # Overrides the BaseCodeExecutor attribute: this executor cannot be stateful.
  stateful: bool = Field(default=False, frozen=True, exclude=True)

  # Overrides the BaseCodeExecutor attribute: this executor cannot
  # optimize_data_file.
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  _client: DockerClient = None
  _container: Container = None

  def __init__(
      self,
      base_url: Optional[str] = None,
      image: Optional[str] = None,
      docker_path: Optional[str] = None,
      **data,
  ):
    """Initializes the ContainerCodeExecutor.

    Args:
      base_url: Optional. The base url of the user hosted Docker client.
      image: The tag of the predefined image or custom image to run on the
        container. Either docker_path or image must be set.
      docker_path: The path to the directory containing the Dockerfile. If set,
        build the image from the dockerfile path instead of using the predefined
        image. Either docker_path or image must be set.
      **data: The data to initialize the ContainerCodeExecutor.
    """
    if not image and not docker_path:
      raise ValueError(
          'Either image or docker_path must be set for ContainerCodeExecutor.'
      )
    if 'stateful' in data and data['stateful']:
      raise ValueError('Cannot set `stateful=True` in ContainerCodeExecutor.')
    if 'optimize_data_file' in data and data['optimize_data_file']:
      raise ValueError(
          'Cannot set `optimize_data_file=True` in ContainerCodeExecutor.'
      )

    super().__init__(**data)
    self.base_url = base_url
    self.image = image if image else DEFAULT_IMAGE_TAG
    self.docker_path = os.path.abspath(docker_path) if docker_path else None

    self._client = (
        docker.from_env()
        if not self.base_url
        else docker.DockerClient(base_url=self.base_url)
    )
    # Initialize the container.
    self.__init_container()

    # Close the container when the on exit.
    atexit.register(self.__cleanup_container)

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    output = ''
    error = ''
    exec_result = self._container.exec_run(
        ['python3', '-c', code_execution_input.code],
        demux=True,
    )

    if exec_result.output and exec_result.output[0]:
      output = exec_result.output[0].decode('utf-8')
    if (
        exec_result.output
        and len(exec_result.output) > 1
        and exec_result.output[1]
    ):
      error = exec_result.output[1].decode('utf-8')

    # Collect the final result.
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )

  def _build_docker_image(self):
    """Builds the Docker image."""
    if not self.docker_path:
      raise ValueError('Docker path is not set.')
    if not os.path.exists(self.docker_path):
      raise FileNotFoundError(f'Invalid Docker path: {self.docker_path}')

    print('Building Docker image...')
    self._client.images.build(
        path=self.docker_path,
        tag=self.image,
        rm=True,
    )
    print(f'Docker image: {self.image} built.')

  def _verify_python_installation(self):
    """Verifies the container has python3 installed."""
    exec_result = self._container.exec_run(['which', 'python3'])
    if exec_result.exit_code != 0:
      raise ValueError('python3 is not installed in the container.')

  def __init_container(self):
    """Initializes the container."""
    if not self._client:
      raise RuntimeError('Docker client is not initialized.')

    if self.docker_path:
      self._build_docker_image()

    print('Starting container for ContainerCodeExecutor...')
    self._container = self._client.containers.run(
        image=self.image,
        detach=True,
        tty=True,
    )
    print(f'Container {self._container.id} started.')

    # Verify the container is able to run python3.
    self._verify_python_installation()

  def __cleanup_container(self):
    """Closes the container on exit."""
    if not self._container:
      return

    print('[Cleanup] Stopping the container...')
    self._container.stop()
    self._container.remove()
    print(f'Container {self._container.id} stopped and removed.')



================================================
FILE: src/google/adk/code_executors/unsafe_local_code_executor.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import redirect_stdout
import io

from pydantic import Field
from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult


class UnsafeLocalCodeExecutor(BaseCodeExecutor):
  """A code executor that unsafely execute code in the current local context."""

  # Overrides the BaseCodeExecutor attribute: this executor cannot be stateful.
  stateful: bool = Field(default=False, frozen=True, exclude=True)

  # Overrides the BaseCodeExecutor attribute: this executor cannot
  # optimize_data_file.
  optimize_data_file: bool = Field(default=False, frozen=True, exclude=True)

  def __init__(self, **data):
    """Initializes the UnsafeLocalCodeExecutor."""
    if 'stateful' in data and data['stateful']:
      raise ValueError('Cannot set `stateful=True` in UnsafeLocalCodeExecutor.')
    if 'optimize_data_file' in data and data['optimize_data_file']:
      raise ValueError(
          'Cannot set `optimize_data_file=True` in UnsafeLocalCodeExecutor.'
      )
    super().__init__(**data)

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    # Execute the code.
    output = ''
    error = ''
    try:
      globals_ = {}
      locals_ = {}
      stdout = io.StringIO()
      with redirect_stdout(stdout):
        exec(code_execution_input.code, globals_, locals_)
      output = stdout.getvalue()
    except Exception as e:
      error = str(e)

    # Collect the final result.
    return CodeExecutionResult(
        stdout=output,
        stderr=error,
        output_files=[],
    )

================================================
FILE: src/google/adk/code_executors/vertex_ai_code_executor.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import mimetypes
import os
from typing import Any, Optional

from typing_extensions import override
from vertexai.preview.extensions import Extension

from ..agents.invocation_context import InvocationContext
from .base_code_executor import BaseCodeExecutor
from .code_execution_utils import CodeExecutionInput
from .code_execution_utils import CodeExecutionResult
from .code_execution_utils import File

_SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg']
_SUPPORTED_DATA_FILE_TYPES = ['csv']

_IMPORTED_LIBRARIES = '''
import io
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

def crop(s: str, max_chars: int = 64) -> str:
  """Crops a string to max_chars characters."""
  return s[: max_chars - 3] + '...' if len(s) > max_chars else s


def explore_df(df: pd.DataFrame) -> None:
  """Prints some information about a pandas DataFrame."""

  with pd.option_context(
      'display.max_columns', None, 'display.expand_frame_repr', False
  ):
    # Print the column names to never encounter KeyError when selecting one.
    df_dtypes = df.dtypes

    # Obtain information about data types and missing values.
    df_nulls = (len(df) - df.isnull().sum()).apply(
        lambda x: f'{x} / {df.shape[0]} non-null'
    )

    # Explore unique total values in columns using `.unique()`.
    df_unique_count = df.apply(lambda x: len(x.unique()))

    # Explore unique values in columns using `.unique()`.
    df_unique = df.apply(lambda x: crop(str(list(x.unique()))))

    df_info = pd.concat(
        (
            df_dtypes.rename('Dtype'),
            df_nulls.rename('Non-Null Count'),
            df_unique_count.rename('Unique Values Count'),
            df_unique.rename('Unique Values'),
        ),
        axis=1,
    )
    df_info.index.name = 'Columns'
    print(f"""Total rows: {df.shape[0]}
Total columns: {df.shape[1]}

{df_info}""")
'''


def _get_code_interpreter_extension(resource_name: str = None):
  """Returns: Load or create the code interpreter extension."""
  if not resource_name:
    resource_name = os.environ.get('CODE_INTERPRETER_EXTENSION_NAME')
  if resource_name:
    new_code_interpreter = Extension(resource_name)
  else:
    print('No CODE_INTERPRETER_ID found in the environment. Create a new one.')
    new_code_interpreter = Extension.from_hub('code_interpreter')
    os.environ['CODE_INTERPRETER_EXTENSION_NAME'] = (
        new_code_interpreter.gca_resource.name
    )
  return new_code_interpreter


class VertexAiCodeExecutor(BaseCodeExecutor):
  """A code executor that uses Vertex Code Interpreter Extension to execute code.

  Attributes:
    resource_name: If set, load the existing resource name of the code
      interpreter extension instead of creating a new one. Format:
      projects/123/locations/us-central1/extensions/456
  """

  resource_name: str = None
  """
  If set, load the existing resource name of the code interpreter extension
  instead of creating a new one.
  Format: projects/123/locations/us-central1/extensions/456
  """

  _code_interpreter_extension: Extension

  def __init__(
      self,
      resource_name: str = None,
      **data,
  ):
    """Initializes the VertexAiCodeExecutor.

    Args:
      resource_name: If set, load the existing resource name of the code
        interpreter extension instead of creating a new one. Format:
        projects/123/locations/us-central1/extensions/456
      **data: Additional keyword arguments to be passed to the base class.
    """
    super().__init__(**data)
    self.resource_name = resource_name
    self._code_interpreter_extension = _get_code_interpreter_extension(
        self.resource_name
    )

  @override
  def execute_code(
      self,
      invocation_context: InvocationContext,
      code_execution_input: CodeExecutionInput,
  ) -> CodeExecutionResult:
    # Execute the code.
    code_execution_result = self._execute_code_interpreter(
        self._get_code_with_imports(code_execution_input.code),
        code_execution_input.input_files,
        code_execution_input.execution_id,
    )

    # Save output file as artifacts.
    current_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name_prefix = '%s_' % str(current_timestamp)
    saved_files = []
    file_count = 0
    for output_file in code_execution_result['output_files']:
      file_type = output_file['name'].split('.')[-1]
      file_name = file_name_prefix + '%d.%s' % (file_count, file_type)
      if file_type in _SUPPORTED_IMAGE_TYPES:
        file_count += 1
        saved_files.append(
            File(
                name='plot_' + file_name,
                content=output_file['contents'],
                mime_type=f'image/{file_type}',
            )
        )
      elif file_type in _SUPPORTED_DATA_FILE_TYPES:
        file_count += 1
        saved_files.append(
            File(
                name='data_' + file_name,
                content=output_file['contents'],
                mime_type=f'text/{file_type}',
            )
        )
      else:
        mime_type, _ = mimetypes.guess_type(file_name)
        saved_files.append(
            File(
                name=file_name,
                content=output_file['contents'],
                mime_type=mime_type,
            )
        )

    # Collect the final result.
    return CodeExecutionResult(
        stdout=code_execution_result.get('execution_result', ''),
        stderr=code_execution_result.get('execution_error', ''),
        output_files=saved_files,
    )

  def _execute_code_interpreter(
      self,
      code: str,
      input_files: Optional[list[File]] = None,
      session_id: Optional[str] = None,
  ) -> dict[str, Any]:
    """Executes the code interpreter extension.

    Args:
      code: The code to execute.
      input_files: The input files to execute the code with.
      session_id: The session ID to execute the code with.

    Returns:
      The response from the code interpreter extension.
    """
    operation_params = {'code': code}
    if input_files:
      operation_params['files'] = [
          {'name': f.name, 'contents': f.content} for f in input_files
      ]
    if session_id:
      operation_params['session_id'] = session_id
    response = self._code_interpreter_extension.execute(
        operation_id='execute',
        operation_params=operation_params,
    )
    return response

  def _get_code_with_imports(self, code: str) -> str:
    """Builds the code string with built-in imports.

    Args:
      code: The code to execute.

    Returns:
      The code string with built-in imports.
    """
    return f"""
{_IMPORTED_LIBRARIES}

{code}
"""



================================================
FILE: src/google/adk/evaluation/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

logger = logging.getLogger(__name__)

__all__ = []

try:
  from .agent_evaluator import AgentEvaluator

  __all__.append('AgentEvaluator')
except ImportError:
  logger.debug(
      'The Vertex[eval] sdk is not installed. If you want to use the Vertex'
      ' Evaluation with agents, please install it(pip install'
      ' "google-cloud-aiplatform[evaluation]). If not, you can ignore this'
      ' warning.'
  )



================================================
FILE: src/google/adk/evaluation/agent_evaluator.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from os import path
from typing import Dict
from typing import List
from typing import Union

from .evaluation_generator import EvaluationGenerator
from .response_evaluator import ResponseEvaluator
from .trajectory_evaluator import TrajectoryEvaluator

# Constants for default runs and evaluation criteria
NUM_RUNS = 2
TOOL_TRAJECTORY_SCORE_KEY = "tool_trajectory_avg_score"
# This evaluation is not very stable.
# This is always optional unless explicitly specified.
RESPONSE_EVALUATION_SCORE_KEY = "response_evaluation_score"
RESPONSE_MATCH_SCORE_KEY = "response_match_score"

ALLOWED_CRITERIA = [
    TOOL_TRAJECTORY_SCORE_KEY,
    RESPONSE_EVALUATION_SCORE_KEY,
    RESPONSE_MATCH_SCORE_KEY,
]


QUERY_COLUMN = "query"
REFERENCE_COLUMN = "reference"
EXPECTED_TOOL_USE_COLUMN = "expected_tool_use"


DEFAULT_CRITERIA = {
    TOOL_TRAJECTORY_SCORE_KEY: 1.0,  # 1-point scale; 1.0 is perfect.
    RESPONSE_MATCH_SCORE_KEY: 0.8,  # Rouge-1 text match; 0.8 is default.
}


def load_json(file_path: str) -> Union[Dict, List]:
  with open(file_path, "r") as f:
    return json.load(f)


class AgentEvaluator:
  """An evaluator for Agents, mainly intended for helping with test cases."""

  @staticmethod
  def find_config_for_test_file(test_file: str):
    """Find the test_config.json file in the same folder as the test file."""
    test_folder = os.path.dirname(test_file)
    config_path = os.path.join(test_folder, "test_config.json")
    if os.path.exists(config_path):
      config_data = load_json(config_path)
      if "criteria" in config_data and isinstance(
          config_data["criteria"], dict
      ):
        return config_data["criteria"]
      else:
        raise ValueError(
            f"Invalid format for test_config.json at {config_path}. Expected a"
            " 'criteria' dictionary."
        )
    return DEFAULT_CRITERIA

  @staticmethod
  def evaluate(
      agent_module,
      eval_dataset_file_path_or_dir,
      num_runs=NUM_RUNS,
      agent_name=None,
      initial_session_file=None,
  ):
    """Evaluates an Agent given eval data.

    Args:
      agent_module: The path to python module that contains the definition of
        the agent. There is convention in place here, where the code is going to
        look for 'root_agent' in the loaded module.
      eval_dataset: The eval data set. This can be either a string representing
        full path to the file containing eval dataset, or a directory that is
        recursively explored for all files that have a `.test.json` suffix.
      num_runs: Number of times all entries in the eval dataset should be
        assessed.
      agent_name: The name of the agent.
      initial_session_file: File that contains initial session state that is
        needed by all the evals in the eval dataset.
    """
    test_files = []
    if isinstance(eval_dataset_file_path_or_dir, str) and os.path.isdir(
        eval_dataset_file_path_or_dir
    ):
      for root, _, files in os.walk(eval_dataset_file_path_or_dir):
        for file in files:
          if file.endswith(".test.json"):
            test_files.append(path.join(root, file))
    else:
      test_files = [eval_dataset_file_path_or_dir]

    initial_session_state = {}
    if initial_session_file:
      with open(initial_session_file, "r") as f:
        initial_session_state = json.loads(f.read())["state"]

    for test_file in test_files:
      dataset = AgentEvaluator._load_dataset(test_file)[0]
      criteria = AgentEvaluator.find_config_for_test_file(test_file)

      AgentEvaluator._validate_input([dataset], criteria)

      evaluation_response = AgentEvaluator._generate_responses(
          agent_module,
          [dataset],
          num_runs,
          agent_name=agent_name,
          initial_session={"state": initial_session_state},
      )

      if AgentEvaluator._response_evaluation_required(criteria, [dataset]):
        AgentEvaluator._evaluate_response_scores(
            agent_module, evaluation_response, criteria
        )

      if AgentEvaluator._trajectory_evaluation_required(criteria, [dataset]):
        AgentEvaluator._evaluate_tool_trajectory(
            agent_module, evaluation_response, criteria
        )

  @staticmethod
  def _load_dataset(
      input_data: Union[str, List[str], List[Dict], List[List[Dict]]],
  ) -> List[List[Dict]]:
    def load_json_file(file_path: str) -> List[Dict]:
      data = load_json(file_path)
      if not isinstance(data, list) or not all(
          isinstance(d, dict) for d in data
      ):
        raise ValueError(f"{file_path} must contain a list of dictionaries.")
      return data

    if isinstance(input_data, str):
      if os.path.isdir(input_data):
        test_files = []
        for root, _, files in os.walk(input_data):
          for file in files:
            if file.endswith(".test.json"):
              test_files.append(os.path.join(root, file))
        return [load_json_file(f) for f in test_files]
      elif os.path.isfile(input_data):
        return [load_json_file(input_data)]
      else:
        raise ValueError(f"Input path {input_data} is invalid.")
    elif isinstance(input_data, list):
      if all(isinstance(i, str) and os.path.isfile(i) for i in input_data):
        return [load_json_file(i) for i in input_data]
      raise TypeError("Input list must contain valid file paths.")
    raise TypeError("Invalid input type for dataset loading.")

  @staticmethod
  def _validate_input(eval_dataset, criteria):
    """Validates that the evaluation criteria align with the provided dataset.

    For efficiency, we only use first row to validate input.
    """
    if not eval_dataset:
      raise ValueError("The evaluation dataset is None or empty.")

    for key in criteria:
      if key not in ALLOWED_CRITERIA:
        raise ValueError(
            f"Invalid criteria key: {key}. Expected one of {ALLOWED_CRITERIA}."
        )

    if not eval_dataset:
      raise ValueError("The evaluation dataset is empty.")
    sample = eval_dataset[0]
    first_query = sample[0]

    if not isinstance(sample, list) and not isinstance(first_query, dict):
      raise ValueError(
          "Each evaluation dataset sample must be list of dictionary. But it's"
          f" {eval_dataset}"
      )

    if TOOL_TRAJECTORY_SCORE_KEY in criteria:
      if (
          QUERY_COLUMN not in first_query
          or EXPECTED_TOOL_USE_COLUMN not in first_query
      ):
        raise ValueError(
            f"Samples for {TOOL_TRAJECTORY_SCORE_KEY} must include"
            f" '{QUERY_COLUMN}' and '{EXPECTED_TOOL_USE_COLUMN}' keys. The"
            f" sample is {sample}."
        )

    if RESPONSE_EVALUATION_SCORE_KEY in criteria:
      if QUERY_COLUMN not in first_query:
        raise ValueError(
            f"Samples for {RESPONSE_EVALUATION_SCORE_KEY} must include"
            f" '{QUERY_COLUMN}' key. The sample is {sample}."
        )

    if RESPONSE_MATCH_SCORE_KEY in criteria:
      if QUERY_COLUMN not in first_query or REFERENCE_COLUMN not in first_query:
        raise ValueError(
            f"Samples for {RESPONSE_MATCH_SCORE_KEY} must include"
            f" '{QUERY_COLUMN}' and '{REFERENCE_COLUMN}' keys. The sample is"
            f" {sample}."
        )

  @staticmethod
  def _get_infer_criteria(eval_dataset):
    """Infers evaluation criteria based on the provided dataset.

    Args:
        eval_dataset (list): A list of evaluation samples.

    Returns:
        dict: Inferred evaluation criteria based on dataset fields.
    """
    inferred_criteria = {}
    sample = eval_dataset[0][0]

    if QUERY_COLUMN in sample and EXPECTED_TOOL_USE_COLUMN in sample:
      inferred_criteria[TOOL_TRAJECTORY_SCORE_KEY] = DEFAULT_CRITERIA[
          TOOL_TRAJECTORY_SCORE_KEY
      ]

    if QUERY_COLUMN in sample and REFERENCE_COLUMN in sample:
      inferred_criteria[RESPONSE_MATCH_SCORE_KEY] = DEFAULT_CRITERIA[
          RESPONSE_MATCH_SCORE_KEY
      ]

    return inferred_criteria

  @staticmethod
  def _generate_responses(
      agent_module, eval_dataset, num_runs, agent_name=None, initial_session={}
  ):
    """Generates evaluation responses by running the agent module multiple times."""
    return EvaluationGenerator.generate_responses(
        eval_dataset,
        agent_module,
        repeat_num=num_runs,
        agent_name=agent_name,
        initial_session=initial_session,
    )

  @staticmethod
  def _generate_responses_from_session(eval_dataset, session_path):
    """Generates evaluation responses by running the agent module multiple times."""
    return EvaluationGenerator.generate_responses_from_session(
        session_path, eval_dataset
    )

  @staticmethod
  def _response_evaluation_required(criteria, eval_dataset):
    """Checks if response evaluation are needed."""
    return REFERENCE_COLUMN in eval_dataset[0][0] and any(
        key in criteria
        for key in [RESPONSE_EVALUATION_SCORE_KEY, RESPONSE_MATCH_SCORE_KEY]
    )

  @staticmethod
  def _trajectory_evaluation_required(evaluation_criteria, eval_dataset):
    """Checks if response evaluation are needed."""
    return (
        EXPECTED_TOOL_USE_COLUMN in eval_dataset[0][0]
        and TOOL_TRAJECTORY_SCORE_KEY in evaluation_criteria
    )

  @staticmethod
  def _evaluate_response_scores(agent_module, evaluation_response, criteria):
    """Evaluates response scores and raises an assertion error if they don't meet the criteria."""
    metrics = ResponseEvaluator.evaluate(
        evaluation_response, criteria, print_detailed_results=True
    )

    AgentEvaluator._assert_score(
        metrics,
        "coherence/mean",
        criteria.get(RESPONSE_EVALUATION_SCORE_KEY),
        "Average response evaluation score",
        agent_module,
    )

    AgentEvaluator._assert_score(
        metrics,
        "rouge_1/mean",
        criteria.get(RESPONSE_MATCH_SCORE_KEY),
        "Average response match score",
        agent_module,
    )

  @staticmethod
  def _evaluate_tool_trajectory(agent_module, evaluation_response, criteria):
    """Evaluates tool trajectory scores and raises an assertion error if they don't meet the criteria."""
    score = TrajectoryEvaluator.evaluate(
        evaluation_response, print_detailed_results=True
    )
    AgentEvaluator._assert_score(
        {TOOL_TRAJECTORY_SCORE_KEY: score},
        TOOL_TRAJECTORY_SCORE_KEY,
        criteria[TOOL_TRAJECTORY_SCORE_KEY],
        "Average tool trajectory evaluation score",
        agent_module,
    )

  @staticmethod
  def _assert_score(metrics, metric_key, threshold, description, agent_module):
    """Asserts that a metric meets the specified threshold."""
    if metric_key in metrics:
      actual_score = metrics[metric_key]
      assert actual_score >= threshold, (
          f"{description} for {agent_module} is lower than expected. "
          f"Expected >= {threshold}, but got {actual_score}."
      )



================================================
FILE: src/google/adk/evaluation/evaluation_constants.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class EvalConstants:
  """Holds constants for evaluation file constants."""

  QUERY = "query"
  EXPECTED_TOOL_USE = "expected_tool_use"
  RESPONSE = "response"
  REFERENCE = "reference"
  TOOL_NAME = "tool_name"
  TOOL_INPUT = "tool_input"
  MOCK_TOOL_OUTPUT = "mock_tool_output"



================================================
FILE: src/google/adk/evaluation/evaluation_generator.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import uuid

from google.genai import types

from ..agents.base_agent import BaseAgent
from ..agents.llm_agent import Agent
from ..agents.llm_agent import BeforeToolCallback
from ..agents.llm_agent import LlmAgent
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..runners import Runner
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from .evaluation_constants import EvalConstants


class EvaluationGenerator:
  """Generates evaluation responses for agents."""

  @staticmethod
  def generate_responses(
      eval_dataset,
      agent_module_path,
      repeat_num=3,
      agent_name=None,
      initial_session={},
  ):
    """Returns evaluation responses for the given dataset and agent.

    Args:
      eval_dataset: The dataset that needs to be scraped for responses.
      agent_module_path: Path to the module that contains the root agent.
      repeat_num: Number of time the eval dataset should be repeated. This is
        usually done to remove uncertainty that a single run may bring.
      agent_name: The name of the agent that should be evaluated. This is
        usually the sub-agent.
      initial_session: Initial session for the eval data.
    """
    results = []

    for _ in range(repeat_num):
      for data in eval_dataset:
        results.append(
            EvaluationGenerator._process_query(
                data, agent_module_path, agent_name, initial_session
            )
        )

    return results

  @staticmethod
  def generate_responses_from_session(session_path, eval_dataset):
    """Returns evaluation responses by combining session data with eval data.

    Args:
      session_path: Path to a json file that contains session data.
      eval_dataset: The eval data set that should be combined with the session
        data.
    """
    results = []

    with open(session_path, "r") as f:
      session_data = Session.model_validate_json(f.read())
      print("loaded session", session_path)

    for data in eval_dataset:
      # load session data from session_path
      results.append(
          EvaluationGenerator._process_query_with_session(
              session_data,
              data,
          )
      )

    return results

  @staticmethod
  def _process_query(data, module_name, agent_name=None, initial_session={}):
    """Process a query using the agent and evaluation dataset."""
    module_path = f"{module_name}"
    agent_module = importlib.import_module(module_path)
    root_agent = agent_module.agent.root_agent

    reset_func = getattr(agent_module.agent, "reset_data", None)

    agent_to_evaluate = root_agent
    if agent_name:
      agent_to_evaluate = root_agent.find_agent(agent_name)
      assert agent_to_evaluate, f"Sub-Agent `{agent_name}` not found."

    return EvaluationGenerator._process_query_with_root_agent(
        data, agent_to_evaluate, reset_func, initial_session
    )

  @staticmethod
  def _process_query_with_root_agent(
      data,
      root_agent,
      reset_func,
      initial_session={},
      session_id=None,
      session_service=None,
      artifact_service=None,
  ):
    """Process a query using the agent and evaluation dataset."""

    # we don't know which tools belong to which agent
    # so we just apply to any agents that has certain tool outputs
    all_mock_tools = set()
    for eval_entry in data:
      expected_tool_use = eval_entry.get(EvalConstants.EXPECTED_TOOL_USE, [])
      for expected in expected_tool_use:
        if EvalConstants.MOCK_TOOL_OUTPUT in expected:
          all_mock_tools.add(expected[EvalConstants.TOOL_NAME])

    eval_data_copy = data.copy()
    EvaluationGenerator.apply_before_tool_callback(
        root_agent,
        lambda *args: EvaluationGenerator.before_tool_callback(
            *args, eval_dataset=eval_data_copy
        ),
        all_mock_tools,
    )

    if not session_service:
      session_service = InMemorySessionService()

    app_name = initial_session.get("app_name", "EvaluationGenerator")
    user_id = initial_session.get("user_id", "test_user_id")
    session_id = session_id if session_id else str(uuid.uuid4())

    _ = session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        state=initial_session.get("state", {}),
        session_id=session_id,
    )

    if not artifact_service:
      artifact_service = InMemoryArtifactService()
    runner = Runner(
        app_name=app_name,
        agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )

    # Reset agent state for each query
    if callable(reset_func):
      reset_func()

    responses = data.copy()

    for index, eval_entry in enumerate(responses):
      response = None
      query = eval_entry["query"]
      content = types.Content(role="user", parts=[types.Part(text=query)])
      turn_actual_tool_uses = []

      for event in runner.run(
          user_id=user_id, session_id=session_id, new_message=content
      ):
        if event.is_final_response() and event.content and event.content.parts:
          response = event.content.parts[0].text
        elif event.get_function_calls():
          for call in event.get_function_calls():
            turn_actual_tool_uses.append({
                EvalConstants.TOOL_NAME: call.name,
                EvalConstants.TOOL_INPUT: call.args,
            })

      responses[index]["actual_tool_use"] = turn_actual_tool_uses
      responses[index]["response"] = response

    return responses

  @staticmethod
  def _process_query_with_session(session_data, data):
    """Process the queries using the existing session data without invoking the runner."""
    responses = data.copy()

    # Iterate through the provided queries and align them with the session events
    for index, eval_entry in enumerate(responses):
      query = eval_entry["query"]
      actual_tool_uses = []
      response = None

      # Search for the corresponding session events
      for event in session_data.events:
        # Match the query to a user event
        if (
            event.author == "user"
            and event.content
            and event.content.parts
            and event.content.parts[0].text == query
        ):
          # Look for subsequent tool usage or model responses
          for subsequent_event in session_data.events:
            if subsequent_event.invocation_id == event.invocation_id:
              # Extract tool usage
              if subsequent_event.content.parts[0].function_call:
                call = subsequent_event.content.parts[0].function_call
                actual_tool_uses.append(
                    {"tool_name": call.name, "tool_input": call.args}
                )
              # Extract final response
              elif subsequent_event.author != "user":
                response = subsequent_event.content.parts[0].text

      # Update the results for the current query
      responses[index]["actual_tool_use"] = actual_tool_uses
      responses[index]["response"] = response
    return responses

  @staticmethod
  def before_tool_callback(tool, args, tool_context, eval_dataset):
    """Intercept specific tool calls and return predefined outputs

    from eval_dataset.
    """
    for index, eval_entry in enumerate(eval_dataset):
      expected_tool_use = eval_entry.get("expected_tool_use", [])
      for expected in expected_tool_use:
        if (
            EvalConstants.MOCK_TOOL_OUTPUT in expected
            and tool.name == expected[EvalConstants.TOOL_NAME]
            and args == expected.get(EvalConstants.TOOL_INPUT, {})
        ):
          # pop the matched entry so we don't rematch again
          eval_dataset.pop(index)
          return {"result": expected[EvalConstants.MOCK_TOOL_OUTPUT]}

    return None

  @staticmethod
  def apply_before_tool_callback(
      agent: BaseAgent,
      callback: BeforeToolCallback,
      all_mock_tools: set[str],
  ):
    """Recursively apply the before_tool_callback to the root agent and all its subagents."""
    # Check if the agent has tools that are defined by evalset.
    # We use function names to check if tools match
    if not isinstance(agent, Agent) and not isinstance(agent, LlmAgent):
      return

    for tool in agent.canonical_tools:
      tool_name = tool.name
      if tool_name in all_mock_tools:
        agent.before_tool_callback = callback

    # Apply recursively to subagents if they exist
    for sub_agent in agent.sub_agents:
      EvaluationGenerator.apply_before_tool_callback(
          sub_agent, callback, all_mock_tools
      )



================================================
FILE: src/google/adk/evaluation/response_evaluator.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import pandas as pd
from tabulate import tabulate
from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation import MetricPromptTemplateExamples


class ResponseEvaluator:
  """Runs response evaluation for agents."""

  @staticmethod
  def evaluate(
      raw_eval_dataset: list[list[dict[str, Any]]],
      evaluation_criteria: list[str],
      *,
      print_detailed_results: bool = False,
  ):
    r"""Returns the value of requested evaluation metrics.

    Args:
      raw_eval_dataset: The dataset that will be evaluated.
      evaluation_criteria: The evaluation criteria to be used. This method
        support two criteria, `response_evaluation_score` and
        `response_match_score`.
      print_detailed_results: Prints detailed results on the console. This is
        usually helpful during debugging.

    A note on evaluation_criteria:
      `response_match_score`: This metric compares the agents final natural
        language response with the expected final response, stored in the
        "reference" field in test/eval files. We use Rouge metric to compare the
        two responses.

        Value Range: [0, 1]. A score closer to 0 means poor similarity between
          response and reference. A score closer to 1 means strong similarity
          between response and reference.

      `response_evaluation_score`: Uses LLM to evalaute coherence of the
        response, including tool use. This is pointwise metric.

        Value range: [0, 5], where 0 means that the agent's response is not
        coherent, while 5 means it is . High values are good.
    A note on raw_eval_dataset:
      The dataset should be a list session, where each session is represented
      as a list of interaction that need evaluation. Each evaluation is
      represented as a dictionary that is expected to have values for the
      following keys:

        1) query
        2) response
        3) acutal_tool_use
        4) expected_tool_use
        5) reference

      Here is a sample eval_dataset value with one entry:
      [
        [
          {
            "query": "roll a die for me",
            "response": "I rolled a 16 sided die and got 13.\n",
            "expected_tool_use": [
              {
                "tool_name": "roll_die",
                "tool_input": {
                  "sides": 16
                }
              }
            ],
            "acutal_tool_use": [
              {
                "tool_name": "roll_die",
                "tool_input": {
                  "sides": 16
                }
              }
            ],
            "reference": "I rolled a 16 sided die and got 13.\n"
          }
        ]
      ]
    """
    if not raw_eval_dataset:
      raise ValueError("The evaluation dataset is empty.")

    metrics = ResponseEvaluator._get_metrics(
        raw_eval_dataset, evaluation_criteria
    )
    flattened_queries = [
        item for sublist in raw_eval_dataset for item in sublist
    ]
    eval_dataset = pd.DataFrame(flattened_queries).rename(
        columns={"query": "prompt", "expected_tool_use": "reference_trajectory"}
    )

    eval_result = ResponseEvaluator._perform_eval(
        dataset=eval_dataset, metrics=metrics
    )

    if print_detailed_results:
      ResponseEvaluator._print_results(eval_result)
    return eval_result.summary_metrics

  @staticmethod
  def _get_metrics(raw_eval_dataset, criteria):
    metrics = []
    if (
        "response_evaluation_score" in criteria
        and "query" in raw_eval_dataset[0][0]
        and "expected_tool_use" in raw_eval_dataset[0][0]
    ):
      metrics.append(MetricPromptTemplateExamples.Pointwise.COHERENCE)
    if (
        "response_match_score" in criteria
        and "reference" in raw_eval_dataset[0][0]
    ):
      metrics.append("rouge_1")
    return metrics

  @staticmethod
  def _perform_eval(dataset, metrics):
    """This method hides away the call to external service.

    Primarily helps with unit testing.
    """
    eval_task = EvalTask(dataset=dataset, metrics=metrics)

    return eval_task.evaluate()

  @staticmethod
  def _print_results(eval_result):
    print("Evaluation Summary Metrics:", eval_result.summary_metrics)
    print(tabulate(eval_result.metrics_table, headers="keys", tablefmt="grid"))



================================================
FILE: src/google/adk/evaluation/trajectory_evaluator.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import pandas as pd
from tabulate import tabulate

from .evaluation_constants import EvalConstants


class TrajectoryEvaluator:
  """Evaluates tool use trajectories for accuracy."""

  @staticmethod
  def evaluate(
      eval_dataset: list[list[dict[str, Any]]],
      *,
      print_detailed_results: bool = False,
  ):
    r"""Returns the mean tool use accuracy of the eval dataset.

    Tool use accuracy is calculated by comparing the expected and the actual
    tool use trajectories. An exact match scores a 1, 0 otherwise. The final
    number is an average of these individual scores.

    Value range: [0, 1], where 0 is means none of the too use entries aligned,
    and 1 would mean all of them aligned. Higher value is good.

    Args:
      eval_dataset: The dataset that will be evaluated.
      print_detailed_results: Prints detailed results on the console. This is
        usually helpful during debugging.

    A note on eval_dataset:
      The dataset should be a list session, where each session is represented
      as a list of interaction that need evaluation. Each evaluation is
      represented as a dictionary that is expected to have values for the
      following keys:
        1) query
        2) response
        3) acutal_tool_use
        4) expected_tool_use

      Here is a sample eval_dataset value with one entry:

      [
        [
          {
            "query": "Roll a 16 sided dice for me",
            "response": "I rolled a 16 sided die and got 13.\n",
            "expected_tool_use": [
              {
                "tool_name": "roll_die",
                "tool_input": {
                  "sides": 16
                }
              }
            ],
            "acutal_tool_use": [
              {
                "tool_name": "roll_die",
                "tool_input": {
                  "sides": 16
                }
              }
            ]
          }
        ]
      ]
    """
    if not eval_dataset:
      raise ValueError("The evaluation dataset is empty.")

    results_df = pd.DataFrame(
        columns=[
            "query",
            "response",
            "actual_tool_use",
            "expected_tool_use",
            "tool_use_accuracy",
        ]
    )
    failures = []

    for conversation in eval_dataset:
      for index, row in enumerate(conversation):
        new_row, failure = TrajectoryEvaluator._evaluate_row(row)
        results_df = pd.concat(
            [results_df, pd.DataFrame([new_row])], ignore_index=True
        )
        if failure:
          failure["turn"] = index + 1
          failures.append(failure)

    TrajectoryEvaluator._report_failures(failures)

    if print_detailed_results:
      TrajectoryEvaluator._print_results(results_df)

    return results_df["tool_use_accuracy"].mean()

  @staticmethod
  def _evaluate_row(row):
    # We don't evaluate the mock tool outputs.
    expected = TrajectoryEvaluator._remove_tool_outputs(
        row["expected_tool_use"]
    )
    actual = row["actual_tool_use"]
    tool_use_accuracy = (
        1.0 if TrajectoryEvaluator.are_tools_equal(actual, expected) else 0.0
    )

    new_row = {
        "query": row["query"],
        "response": row["response"],
        "actual_tool_use": actual,
        "expected_tool_use": expected,
        "tool_use_accuracy": tool_use_accuracy,
    }
    failure = (
        None
        if tool_use_accuracy == 1.0
        else {"query": row["query"], "actual": actual, "expected": expected}
    )
    return new_row, failure

  @staticmethod
  def are_tools_equal(list_a_original, list_b_original):
    # Remove other entries that we don't want to evaluate
    list_a = [
        {"tool_name": tool["tool_name"], "tool_input": tool["tool_input"]}
        for tool in list_a_original
    ]

    list_b = [
        {"tool_name": tool["tool_name"], "tool_input": tool["tool_input"]}
        for tool in list_b_original
    ]

    return list_a == list_b

  @staticmethod
  def _remove_tool_outputs(tool_use_list):
    """Removes 'mock_tool_output' from each dictionary in the list."""
    result = []
    for tool_use in tool_use_list:
      new_tool_use = (
          tool_use.copy()
      )  # Create a copy to avoid modifying the original
      new_tool_use.pop(
          EvalConstants.MOCK_TOOL_OUTPUT, None
      )  # Remove 'tool_output' if it exists
      result.append(new_tool_use)
    return result

  @staticmethod
  def _report_failures(failures):
    if failures:
      print("Failures:")
      for failure in failures:
        print(f"""{{
  "turn": {failure["turn"]},
  "query": '{failure["query"]}',
  "actual": {failure["actual"]},
  "expected_tool_use": {failure["expected"]},
}}
""")

  @staticmethod
  def _print_results(results_df):
    print(tabulate(results_df, headers="keys", tablefmt="grid"))



================================================
FILE: src/google/adk/events/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .event import Event
from .event_actions import EventActions

__all__ = [
    'Event',
    'EventActions',
]



================================================
FILE: src/google/adk/events/event.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from datetime import datetime
import random
import string
from typing import Optional

from google.genai import types
from pydantic import ConfigDict
from pydantic import Field

from ..models.llm_response import LlmResponse
from .event_actions import EventActions


class Event(LlmResponse):
  """Represents an event in a conversation between agents and users.

  It is used to store the content of the conversation, as well as the actions
  taken by the agents like function calls, etc.

  Attributes:
    invocation_id: The invocation ID of the event.
    author: "user" or the name of the agent, indicating who appended the event
      to the session.
    actions: The actions taken by the agent.
    long_running_tool_ids: The ids of the long running function calls.
    branch: The branch of the event.
    id: The unique identifier of the event.
    timestamp: The timestamp of the event.
    is_final_response: Whether the event is the final response of the agent.
    get_function_calls: Returns the function calls in the event.
  """

  model_config = ConfigDict(
      extra='forbid', ser_json_bytes='base64', val_json_bytes='base64'
  )
  """The pydantic model config."""

  # TODO: revert to be required after spark migration
  invocation_id: str = ''
  """The invocation ID of the event."""
  author: str
  """'user' or the name of the agent, indicating who appended the event to the
  session."""
  actions: EventActions = Field(default_factory=EventActions)
  """The actions taken by the agent."""

  long_running_tool_ids: Optional[set[str]] = None
  """Set of ids of the long running function calls.
  Agent client will know from this field about which function call is long running.
  only valid for function call event
  """
  branch: Optional[str] = None
  """The branch of the event.

  The format is like agent_1.agent_2.agent_3, where agent_1 is the parent of
  agent_2, and agent_2 is the parent of agent_3.

  Branch is used when multiple sub-agent shouldn't see their peer agents'
  conversation history.
  """

  # The following are computed fields.
  # Do not assign the ID. It will be assigned by the session.
  id: str = ''
  """The unique identifier of the event."""
  timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
  """The timestamp of the event."""

  def model_post_init(self, __context):
    """Post initialization logic for the event."""
    # Generates a random ID for the event.
    if not self.id:
      self.id = Event.new_id()

  def is_final_response(self) -> bool:
    """Returns whether the event is the final response of the agent."""
    if self.actions.skip_summarization or self.long_running_tool_ids:
      return True
    return (
        not self.get_function_calls()
        and not self.get_function_responses()
        and not self.partial
        and not self.has_trailing_code_execution_result()
    )

  def get_function_calls(self) -> list[types.FunctionCall]:
    """Returns the function calls in the event."""
    func_calls = []
    if self.content and self.content.parts:
      for part in self.content.parts:
        if part.function_call:
          func_calls.append(part.function_call)
    return func_calls

  def get_function_responses(self) -> list[types.FunctionResponse]:
    """Returns the function responses in the event."""
    func_response = []
    if self.content and self.content.parts:
      for part in self.content.parts:
        if part.function_response:
          func_response.append(part.function_response)
    return func_response

  def has_trailing_code_execution_result(
      self,
  ) -> bool:
    """Returns whether the event has a trailing code execution result."""
    if self.content:
      if self.content.parts:
        return self.content.parts[-1].code_execution_result is not None
    return False

  @staticmethod
  def new_id():
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(8))



================================================
FILE: src/google/adk/events/event_actions.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..auth.auth_tool import AuthConfig


class EventActions(BaseModel):
  """Represents the actions attached to an event."""

  model_config = ConfigDict(extra='forbid')
  """The pydantic model config."""

  skip_summarization: Optional[bool] = None
  """If true, it won't call model to summarize function response.

  Only used for function_response event.
  """

  state_delta: dict[str, object] = Field(default_factory=dict)
  """Indicates that the event is updating the state with the given delta."""

  artifact_delta: dict[str, int] = Field(default_factory=dict)
  """Indicates that the event is updating an artifact. key is the filename,
  value is the version."""

  transfer_to_agent: Optional[str] = None
  """If set, the event transfers to the specified agent."""

  escalate: Optional[bool] = None
  """The agent is escalating to a higher level agent."""

  requested_auth_configs: dict[str, AuthConfig] = Field(default_factory=dict)
  """Authentication configurations requested by tool responses.

  This field will only be set by a tool response event indicating tool request
  auth credential.
  - Keys: The function call id. Since one function response event could contain
  multiple function responses that correspond to multiple function calls. Each
  function call could request different auth configs. This id is used to
  identify the function call.
  - Values: The requested auth config.
  """



================================================
FILE: src/google/adk/examples/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base_example_provider import BaseExampleProvider
from .example import Example

__all__ = [
    'BaseExampleProvider',
    'Example',
]

try:
  from .vertex_ai_example_store import VertexAiExampleStore

  __all__.append('VertexAiExampleStore')
except ImportError:
  pass



================================================
FILE: src/google/adk/examples/base_example_provider.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from .example import Example


# A class that provides examples for a given query.
class BaseExampleProvider(abc.ABC):
  """Base class for example providers.

  This class defines the interface for providing examples for a given query.
  """

  @abc.abstractmethod
  def get_examples(self, query: str) -> list[Example]:
    """Returns a list of examples for a given query.

    Args:
        query: The query to get examples for.

    Returns:
        A list of Example objects.
    """



================================================
FILE: src/google/adk/examples/example.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.genai import types
from pydantic import BaseModel


class Example(BaseModel):
  """A few-shot example.

  Attributes:
    input: The input content for the example.
    output: The expected output content for the example.
  """

  input: types.Content
  output: list[types.Content]



================================================
FILE: src/google/adk/examples/example_util.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for converting examples to a string that can be used in system instructions in the prompt."""

import logging
from typing import Optional, Union
from typing import TYPE_CHECKING

from .base_example_provider import BaseExampleProvider
from .example import Example

if TYPE_CHECKING:
  from ..sessions.session import Session

logger = logging.getLogger(__name__)

# Constant parts of the example string
_EXAMPLES_INTRO = (
    "<EXAMPLES>\nBegin few-shot\nThe following are examples of user queries and"
    " model responses using the available tools.\n\n"
)
_EXAMPLES_END = "End few-shot\n<EXAMPLES>"
_EXAMPLE_START = "EXAMPLE {}:\nBegin example\n"
_EXAMPLE_END = "End example\n\n"
_USER_PREFIX = "[user]\n"
_MODEL_PREFIX = "[model]\n"
_FUNCTION_PREFIX = "```\n"
_FUNCTION_CALL_PREFIX = "```tool_code\n"
_FUNCTION_CALL_SUFFIX = "\n```\n"
_FUNCTION_RESPONSE_PREFIX = "```tool_outputs\n"
_FUNCTION_RESPONSE_SUFFIX = "\n```\n"


# TODO(yaojie): Add unit tests for this function.
def convert_examples_to_text(
    examples: list[Example], model: Optional[str]
) -> str:
  """Converts a list of examples to a string that can be used in a system instruction."""
  examples_str = ""
  for example_num, example in enumerate(examples):
    output = f"{_EXAMPLE_START.format(example_num + 1)}{_USER_PREFIX}"
    if example.input and example.input.parts:
      output += (
          "\n".join(part.text for part in example.input.parts if part.text)
          + "\n"
      )

    gemini2 = model is None or "gemini-2" in model
    previous_role = None
    for content in example.output:
      role = _MODEL_PREFIX if content.role == "model" else _USER_PREFIX
      if role != previous_role:
        output += role
      previous_role = role
      for part in content.parts:
        if part.function_call:
          args = []
          # Convert function call part to python-like function call
          for k, v in part.function_call.args.items():
            if isinstance(v, str):
              args.append(f"{k}='{v}'")
            else:
              args.append(f"{k}={v}")
          prefix = _FUNCTION_PREFIX if gemini2 else _FUNCTION_CALL_PREFIX
          output += (
              f"{prefix}{part.function_call.name}({', '.join(args)}){_FUNCTION_CALL_SUFFIX}"
          )
        # Convert function response part to json string
        elif part.function_response:
          prefix = _FUNCTION_PREFIX if gemini2 else _FUNCTION_RESPONSE_PREFIX
          output += f"{prefix}{part.function_response.__dict__}{_FUNCTION_RESPONSE_SUFFIX}"
        elif part.text:
          output += f"{part.text}\n"

    output += _EXAMPLE_END
    examples_str += output

  return f"{_EXAMPLES_INTRO}{examples_str}{_EXAMPLES_END}"


def _get_latest_message_from_user(session: "Session") -> str:
  """Gets the latest message from the user.

  Returns:
    The latest message from the user. If not found, returns an empty string.
  """
  events = session.events
  if not events:
    return ""

  event = events[-1]
  if event.author == "user" and not event.get_function_responses():
    if event.content.parts and event.content.parts[0].text:
      return event.content.parts[0].text
    else:
      logger.warning("No message from user for fetching example.")

  return ""


def build_example_si(
    examples: Union[list[Example], BaseExampleProvider],
    query: str,
    model: Optional[str],
) -> str:
  if isinstance(examples, list):
    return convert_examples_to_text(examples, model)
  if isinstance(examples, BaseExampleProvider):
    return convert_examples_to_text(examples.get_examples(query), model)

  raise ValueError("Invalid example configuration")



================================================
FILE: src/google/adk/examples/vertex_ai_example_store.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.genai import types
from typing_extensions import override
from vertexai.preview import example_stores

from .base_example_provider import BaseExampleProvider
from .example import Example


class VertexAiExampleStore(BaseExampleProvider):
  """Provides examples from Vertex example store."""

  def __init__(self, examples_store_name: str):
    """Initializes the VertexAiExampleStore.

    Args:
        examples_store_name: The resource name of the vertex example store, in
          the format of
          ``projects/{project}/locations/{location}/exampleStores/{example_store}``.
    """
    self.examples_store_name = examples_store_name

  @override
  def get_examples(self, query: str) -> list[Example]:
    example_store = example_stores.ExampleStore(self.examples_store_name)
    # Retrieve relevant examples.
    request = {
        "stored_contents_example_parameters": {
            "content_search_key": {
                "contents": [{"role": "user", "parts": [{"text": query}]}],
                "search_key_generation_method": {"last_entry": {}},
            }
        },
        "top_k": 10,
        "example_store": self.examples_store_name,
    }
    response = example_store.api_client.search_examples(request)

    returned_examples = []
    # Convert results to genai formats
    for result in response.results:
      if result.similarity_score < 0.5:
        continue
      expected_contents = [
          content.content
          for content in result.example.stored_contents_example.contents_example.expected_contents
      ]
      expected_output = []
      for content in expected_contents:
        expected_parts = []
        for part in content.parts:
          if part.text:
            expected_parts.append(types.Part.from_text(text=part.text))
          elif part.function_call:
            expected_parts.append(
                types.Part.from_function_call(
                    name=part.function_call.name,
                    args={
                        key: value
                        for key, value in part.function_call.args.items()
                    },
                )
            )
          elif part.function_response:
            expected_parts.append(
                types.Part.from_function_response(
                    name=part.function_response.name,
                    response={
                        key: value
                        for key, value in part.function_response.response.items()
                    },
                )
            )
        expected_output.append(
            types.Content(role=content.role, parts=expected_parts)
        )

      returned_examples.append(
          Example(
              input=types.Content(
                  role="user",
                  parts=[
                      types.Part.from_text(
                          text=result.example.stored_contents_example.search_key
                      )
                  ],
              ),
              output=expected_output,
          )
      )
    return returned_examples



================================================
FILE: src/google/adk/flows/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



================================================
FILE: src/google/adk/flows/llm_flows/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import _code_execution
from . import _nl_planning
from . import contents
from . import functions
from . import identity
from . import instructions



================================================
FILE: src/google/adk/flows/llm_flows/_base_llm_processor.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the processor interface used for BaseLlmFlow."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from ...agents.invocation_context import InvocationContext
from ...events.event import Event

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest
  from ...models.llm_response import LlmResponse


class BaseLlmRequestProcessor(ABC):
  """Base class for LLM request processor."""

  @abstractmethod
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    """Runs the processor."""
    raise NotImplementedError("Not implemented.")
    yield  # AsyncGenerator requires a yield in function body.


class BaseLlmResponseProcessor(ABC):
  """Base class for LLM response processor."""

  @abstractmethod
  async def run_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    """Processes the LLM response."""
    raise NotImplementedError("Not implemented.")
    yield  # AsyncGenerator requires a yield in function body.



================================================
FILE: src/google/adk/flows/llm_flows/_code_execution.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handles Code Execution related logic."""

from __future__ import annotations

import base64
import copy
import dataclasses
import os
import re
from typing import AsyncGenerator
from typing import Generator
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...code_executors.base_code_executor import BaseCodeExecutor
from ...code_executors.code_execution_utils import CodeExecutionInput
from ...code_executors.code_execution_utils import CodeExecutionResult
from ...code_executors.code_execution_utils import CodeExecutionUtils
from ...code_executors.code_execution_utils import File
from ...code_executors.code_executor_context import CodeExecutorContext
from ...events.event import Event
from ...events.event_actions import EventActions
from ...models.llm_response import LlmResponse
from ._base_llm_processor import BaseLlmRequestProcessor
from ._base_llm_processor import BaseLlmResponseProcessor

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest


@dataclasses.dataclass
class DataFileUtil:
  """A structure that contains a data file name and its content."""

  extension: str
  """
  The file extension (e.g., ".csv").
  """

  loader_code_template: str
  """
  The code template to load the data file.
  """


_DATA_FILE_UTIL_MAP = {
    'text/csv': DataFileUtil(
        extension='.csv',
        loader_code_template="pd.read_csv('{filename}')",
    ),
}

_DATA_FILE_HELPER_LIB = '''
import pandas as pd

def explore_df(df: pd.DataFrame) -> None:
  """Prints some information about a pandas DataFrame."""

  with pd.option_context(
      'display.max_columns', None, 'display.expand_frame_repr', False
  ):
    # Print the column names to never encounter KeyError when selecting one.
    df_dtypes = df.dtypes

    # Obtain information about data types and missing values.
    df_nulls = (len(df) - df.isnull().sum()).apply(
        lambda x: f'{x} / {df.shape[0]} non-null'
    )

    # Explore unique total values in columns using `.unique()`.
    df_unique_count = df.apply(lambda x: len(x.unique()))

    # Explore unique values in columns using `.unique()`.
    df_unique = df.apply(lambda x: crop(str(list(x.unique()))))

    df_info = pd.concat(
        (
            df_dtypes.rename('Dtype'),
            df_nulls.rename('Non-Null Count'),
            df_unique_count.rename('Unique Values Count'),
            df_unique.rename('Unique Values'),
        ),
        axis=1,
    )
    df_info.index.name = 'Columns'
    print(f"""Total rows: {df.shape[0]}
Total columns: {df.shape[1]}

{df_info}""")
'''


class _CodeExecutionRequestProcessor(BaseLlmRequestProcessor):
  """Processes code execution requests."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    if not isinstance(invocation_context.agent, LlmAgent):
      return
    if not invocation_context.agent.code_executor:
      return

    async for event in _run_pre_processor(invocation_context, llm_request):
      yield event

    # Convert the code execution parts to text parts.
    if not isinstance(invocation_context.agent.code_executor, BaseCodeExecutor):
      return
    for content in llm_request.contents:
      CodeExecutionUtils.convert_code_execution_parts(
          content,
          invocation_context.agent.code_executor.code_block_delimiters[0]
          if invocation_context.agent.code_executor.code_block_delimiters
          else ('', ''),
          invocation_context.agent.code_executor.execution_result_delimiters,
      )


request_processor = _CodeExecutionRequestProcessor()


class _CodeExecutionResponseProcessor(BaseLlmResponseProcessor):
  """Processes code execution responses."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    # Skip if the response is partial (streaming).
    if llm_response.partial:
      return

    async for event in _run_post_processor(invocation_context, llm_response):
      yield event


response_processor = _CodeExecutionResponseProcessor()


async def _run_pre_processor(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> AsyncGenerator[Event, None]:
  """Pre-process the user message by adding the user message to the Colab notebook."""
  from ...agents.llm_agent import LlmAgent

  if not isinstance(invocation_context.agent, LlmAgent):
    return

  agent = invocation_context.agent
  code_executor = agent.code_executor

  if not code_executor or not isinstance(code_executor, BaseCodeExecutor):
    return
  if not code_executor.optimize_data_file:
    return

  code_executor_context = CodeExecutorContext(invocation_context.session.state)

  # Skip if the error count exceeds the max retry attempts.
  if (
      code_executor_context.get_error_count(invocation_context.invocation_id)
      >= code_executor.error_retry_attempts
  ):
    return

  # [Step 1] Extract data files from the session_history and store them in
  # memory. Meanwhile, mutate the inline data file to text part in session
  # history from all turns.
  all_input_files = _extrac_and_replace_inline_files(
      code_executor_context, llm_request
  )

  # [Step 2] Run Explore_Df code on the data files from the current turn. We
  # only need to explore the new data files because the previous data files
  # should already be explored and cached in the code execution runtime.
  processed_file_names = set(code_executor_context.get_processed_file_names())
  files_to_process = [
      f for f in all_input_files if f.name not in processed_file_names
  ]
  for file in files_to_process:
    code_str = _get_data_file_preprocessing_code(file)
    # Skip for unsupported file or executor types.
    if not code_str:
      return

    # Emit the code to execute, and add it to the LLM request.
    code_content = types.Content(
        role='model',
        parts=[
            types.Part(text=f'Processing input file: `{file.name}`'),
            CodeExecutionUtils.build_executable_code_part(code_str),
        ],
    )
    llm_request.contents.append(copy.deepcopy(code_content))
    yield Event(
        invocation_id=invocation_context.invocation_id,
        author=agent.name,
        branch=invocation_context.branch,
        content=code_content,
    )

    code_execution_result = code_executor.execute_code(
        invocation_context,
        CodeExecutionInput(
            code=code_str,
            input_files=[file],
            execution_id=_get_or_set_execution_id(
                invocation_context, code_executor_context
            ),
        ),
    )
    # Update the processing results to code executor context.
    code_executor_context.update_code_execution_result(
        invocation_context.invocation_id,
        code_str,
        code_execution_result.stdout,
        code_execution_result.stderr,
    )
    code_executor_context.add_processed_file_names([file.name])

    # Emit the execution result, and add it to the LLM request.
    execution_result_event = await _post_process_code_execution_result(
        invocation_context, code_executor_context, code_execution_result
    )
    yield execution_result_event
    llm_request.contents.append(copy.deepcopy(execution_result_event.content))


async def _run_post_processor(
    invocation_context: InvocationContext,
    llm_response,
) -> AsyncGenerator[Event, None]:
  """Post-process the model response by extracting and executing the first code block."""
  agent = invocation_context.agent
  code_executor = agent.code_executor

  if not code_executor or not isinstance(code_executor, BaseCodeExecutor):
    return
  if not llm_response or not llm_response.content:
    return

  code_executor_context = CodeExecutorContext(invocation_context.session.state)
  # Skip if the error count exceeds the max retry attempts.
  if (
      code_executor_context.get_error_count(invocation_context.invocation_id)
      >= code_executor.error_retry_attempts
  ):
    return

  # [Step 1] Extract code from the model predict response and truncate the
  # content to the part with the first code block.
  response_content = llm_response.content
  code_str = CodeExecutionUtils.extract_code_and_truncate_content(
      response_content, code_executor.code_block_delimiters
  )
  # Terminal state: no code to execute.
  if not code_str:
    return

  # [Step 2] Executes the code and emit 2 Events for code and execution result.
  yield Event(
      invocation_id=invocation_context.invocation_id,
      author=agent.name,
      branch=invocation_context.branch,
      content=response_content,
      actions=EventActions(),
  )

  code_execution_result = code_executor.execute_code(
      invocation_context,
      CodeExecutionInput(
          code=code_str,
          input_files=code_executor_context.get_input_files(),
          execution_id=_get_or_set_execution_id(
              invocation_context, code_executor_context
          ),
      ),
  )
  code_executor_context.update_code_execution_result(
      invocation_context.invocation_id,
      code_str,
      code_execution_result.stdout,
      code_execution_result.stderr,
  )
  yield await _post_process_code_execution_result(
      invocation_context, code_executor_context, code_execution_result
  )

  # [Step 3] Skip processing the original model response
  # to continue code generation loop.
  llm_response.content = None


def _extrac_and_replace_inline_files(
    code_executor_context: CodeExecutorContext,
    llm_request: LlmRequest,
) -> list[File]:
  """Extracts and replaces inline files with file names in the LLM request."""
  all_input_files = code_executor_context.get_input_files()
  saved_file_names = set(f.name for f in all_input_files)

  # [Step 1] Process input files from LlmRequest and cache them in CodeExecutor.
  for i in range(len(llm_request.contents)):
    content = llm_request.contents[i]
    # Only process the user message.
    if content.role != 'user' and not content.parts:
      continue

    for j in range(len(content.parts)):
      part = content.parts[j]
      # Skip if the inline data is not supported.
      if (
          not part.inline_data
          or part.inline_data.mime_type not in _DATA_FILE_UTIL_MAP
      ):
        continue

      # Replace the inline data file with a file name placeholder.
      mime_type = part.inline_data.mime_type
      file_name = f'data_{i+1}_{j+1}' + _DATA_FILE_UTIL_MAP[mime_type].extension
      llm_request.contents[i].parts[j] = types.Part(
          text='\nAvailable file: `%s`\n' % file_name
      )

      # Add the inlne data as input file to the code executor context.
      file = File(
          name=file_name,
          content=CodeExecutionUtils.get_encoded_file_content(
              part.inline_data.data
          ).decode(),
          mime_type=mime_type,
      )
      if file_name not in saved_file_names:
        code_executor_context.add_input_files([file])
        all_input_files.append(file)

  return all_input_files


def _get_or_set_execution_id(
    invocation_context: InvocationContext,
    code_executor_context: CodeExecutorContext,
) -> Optional[str]:
  """Returns the ID for stateful code execution or None if not stateful."""
  if not invocation_context.agent.code_executor.stateful:
    return None

  execution_id = code_executor_context.get_execution_id()
  if not execution_id:
    execution_id = invocation_context.session.id
    code_executor_context.set_execution_id(execution_id)
  return execution_id


async def _post_process_code_execution_result(
    invocation_context: InvocationContext,
    code_executor_context: CodeExecutorContext,
    code_execution_result: CodeExecutionResult,
) -> Event:
  """Post-process the code execution result and emit an Event."""
  if invocation_context.artifact_service is None:
    raise ValueError('Artifact service is not initialized.')

  result_content = types.Content(
      role='model',
      parts=[
          CodeExecutionUtils.build_code_execution_result_part(
              code_execution_result
          ),
      ],
  )
  event_actions = EventActions(
      state_delta=code_executor_context.get_state_delta()
  )

  # Handle code execution error retry.
  if code_execution_result.stderr:
    code_executor_context.increment_error_count(
        invocation_context.invocation_id
    )
  else:
    code_executor_context.reset_error_count(invocation_context.invocation_id)

  # Handle output files.
  for output_file in code_execution_result.output_files:
    version = await invocation_context.artifact_service.save_artifact(
        app_name=invocation_context.app_name,
        user_id=invocation_context.user_id,
        session_id=invocation_context.session.id,
        filename=output_file.name,
        artifact=types.Part.from_bytes(
            data=base64.b64decode(output_file.content),
            mime_type=output_file.mime_type,
        ),
    )
    event_actions.artifact_delta[output_file.name] = version

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      branch=invocation_context.branch,
      content=result_content,
      actions=event_actions,
  )


def _get_data_file_preprocessing_code(file: File) -> Optional[str]:
  """Returns the code to explore the data file."""

  def _get_normalized_file_name(file_name: str) -> str:
    var_name, _ = os.path.splitext(file_name)
    # Replace non-alphanumeric characters with underscores
    var_name = re.sub(r'[^a-zA-Z0-9_]', '_', var_name)

    # If the filename starts with a digit, prepend an underscore
    if var_name[0].isdigit():
      var_name = '_' + var_name
    return var_name

  if file.mime_type not in _DATA_FILE_UTIL_MAP:
    return

  var_name = _get_normalized_file_name(file.name)
  loader_code = _DATA_FILE_UTIL_MAP[file.mime_type].loader_code_template.format(
      filename=file.name
  )
  return f"""
{_DATA_FILE_HELPER_LIB}

# Load the dataframe.
{var_name} = {loader_code}

# Use `explore_df` to guide my analysis.
explore_df({var_name})
"""



================================================
FILE: src/google/adk/flows/llm_flows/_nl_planning.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handles NL planning related logic."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import Generator
from typing import Optional
from typing import TYPE_CHECKING

from typing_extensions import override

from ...agents.callback_context import CallbackContext
from ...agents.invocation_context import InvocationContext
from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...planners.plan_re_act_planner import PlanReActPlanner
from ._base_llm_processor import BaseLlmRequestProcessor
from ._base_llm_processor import BaseLlmResponseProcessor

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest
  from ...models.llm_response import LlmResponse
  from ...planners.base_planner import BasePlanner
  from ...planners.built_in_planner import BuiltInPlanner


class _NlPlanningRequestProcessor(BaseLlmRequestProcessor):
  """Processor for NL planning."""

  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...planners.built_in_planner import BuiltInPlanner

    planner = _get_planner(invocation_context)
    if not planner:
      return

    if isinstance(planner, BuiltInPlanner):
      planner.apply_thinking_config(llm_request)

    planning_instruction = planner.build_planning_instruction(
        ReadonlyContext(invocation_context), llm_request
    )
    if planning_instruction:
      llm_request.append_instructions([planning_instruction])

    _remove_thought_from_request(llm_request)

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _NlPlanningRequestProcessor()


class _NlPlanningResponse(BaseLlmResponseProcessor):

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    if (
        not llm_response
        or not llm_response.content
        or not llm_response.content.parts
    ):
      return

    planner = _get_planner(invocation_context)
    if not planner:
      return

    # Postprocess the LLM response.
    callback_context = CallbackContext(invocation_context)
    processed_parts = planner.process_planning_response(
        callback_context, llm_response.content.parts
    )
    if processed_parts:
      llm_response.content.parts = processed_parts

    if callback_context.state.has_delta():
      state_update_event = Event(
          invocation_id=invocation_context.invocation_id,
          author=invocation_context.agent.name,
          branch=invocation_context.branch,
          actions=callback_context._event_actions,
      )
      yield state_update_event


response_processor = _NlPlanningResponse()


def _get_planner(
    invocation_context: InvocationContext,
) -> Optional[BasePlanner]:
  from ...agents.llm_agent import Agent
  from ...planners.base_planner import BasePlanner

  agent = invocation_context.agent
  if not isinstance(agent, Agent):
    return None
  if not agent.planner:
    return None

  if isinstance(agent.planner, BasePlanner):
    return agent.planner
  return PlanReActPlanner()


def _remove_thought_from_request(llm_request: LlmRequest):
  if not llm_request.contents:
    return

  for content in llm_request.contents:
    if not content.parts:
      continue
    for part in content.parts:
      part.thought = None



================================================
FILE: src/google/adk/flows/llm_flows/agent_transfer.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handles agent transfer for LLM flow."""

from __future__ import annotations

import typing
from typing import AsyncGenerator

from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ...tools.function_tool import FunctionTool
from ...tools.tool_context import ToolContext
from ...tools.transfer_to_agent_tool import transfer_to_agent
from ._base_llm_processor import BaseLlmRequestProcessor

if typing.TYPE_CHECKING:
  from ...agents import BaseAgent
  from ...agents import LlmAgent


class _AgentTransferLlmRequestProcessor(BaseLlmRequestProcessor):
  """Agent transfer request processor."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    if not isinstance(invocation_context.agent, LlmAgent):
      return

    transfer_targets = _get_transfer_targets(invocation_context.agent)
    if not transfer_targets:
      return

    llm_request.append_instructions([
        _build_target_agents_instructions(
            invocation_context.agent, transfer_targets
        )
    ])

    transfer_to_agent_tool = FunctionTool(func=transfer_to_agent)
    tool_context = ToolContext(invocation_context)
    await transfer_to_agent_tool.process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    return
    yield  # AsyncGenerator requires yield statement in function body.


request_processor = _AgentTransferLlmRequestProcessor()


def _build_target_agents_info(target_agent: BaseAgent) -> str:
  return f"""
Agent name: {target_agent.name}
Agent description: {target_agent.description}
"""


line_break = '\n'


def _build_target_agents_instructions(
    agent: LlmAgent, target_agents: list[BaseAgent]
) -> str:
  si = f"""
You have a list of other agents to transfer to:

{line_break.join([
    _build_target_agents_info(target_agent) for target_agent in target_agents
])}

If you are the best to answer the question according to your description, you
can answer it.

If another agent is better for answering the question according to its
description, call `{_TRANSFER_TO_AGENT_FUNCTION_NAME}` function to transfer the
question to that agent. When transferring, do not generate any text other than
the function call.
"""

  if agent.parent_agent:
    si += f"""
Your parent agent is {agent.parent_agent.name}. If neither the other agents nor
you are best for answering the question according to the descriptions, transfer
to your parent agent. If you don't have parent agent, try answer by yourself.
"""
  return si


_TRANSFER_TO_AGENT_FUNCTION_NAME = transfer_to_agent.__name__


def _get_transfer_targets(agent: LlmAgent) -> list[BaseAgent]:
  from ...agents.llm_agent import LlmAgent

  result = []
  result.extend(agent.sub_agents)

  if not agent.parent_agent or not isinstance(agent.parent_agent, LlmAgent):
    return result

  if not agent.disallow_transfer_to_parent:
    result.append(agent.parent_agent)

  if not agent.disallow_transfer_to_peers:
    result.extend([
        peer_agent
        for peer_agent in agent.parent_agent.sub_agents
        if peer_agent.name != agent.name
    ])

  return result



================================================
FILE: src/google/adk/flows/llm_flows/audio_transcriber.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import TYPE_CHECKING

from google.cloud import speech
from google.genai import types as genai_types

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext


class AudioTranscriber:
  """Transcribes audio using Google Cloud Speech-to-Text."""

  def __init__(self):
    self.client = speech.SpeechClient()

  def transcribe_file(
      self, invocation_context: InvocationContext
  ) -> list[genai_types.Content]:
    """Transcribe audio, bundling consecutive segments from the same speaker.

    The ordering of speakers will be preserved. Audio blobs will be merged for
    the same speaker as much as we can do reduce the transcription latency.

    Args:
        invocation_context: The invocation context to access the transcription
          cache.

    Returns:
        A list of Content objects containing the transcribed text.
    """

    bundled_audio = []
    current_speaker = None
    current_audio_data = b''
    contents = []

    # Step1: merge audio blobs
    for transcription_entry in invocation_context.transcription_cache or []:
      speaker, audio_data = (
          transcription_entry.role,
          transcription_entry.data,
      )

      if isinstance(audio_data, genai_types.Content):
        if current_speaker is not None:
          bundled_audio.append((current_speaker, current_audio_data))
          current_speaker = None
          current_audio_data = b''
        bundled_audio.append((speaker, audio_data))
        continue

      if not audio_data.data:
        continue

      if speaker == current_speaker:
        current_audio_data += audio_data.data
      else:
        if current_speaker is not None:
          bundled_audio.append((current_speaker, current_audio_data))
        current_speaker = speaker
        current_audio_data = audio_data.data

    # Append the last audio segment if any
    if current_speaker is not None:
      bundled_audio.append((current_speaker, current_audio_data))

    # reset cache
    invocation_context.transcription_cache = []

    # Step2: transcription
    for speaker, data in bundled_audio:
      if speaker == 'user':
        audio = speech.RecognitionAudio(content=data)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US',
        )

        response = self.client.recognize(config=config, audio=audio)

        for result in response.results:
          transcript = result.alternatives[0].transcript

          parts = [genai_types.Part(text=transcript)]
          role = speaker.lower()
          content = genai_types.Content(role=role, parts=parts)
          contents.append(content)
      else:
        # don't need to transcribe model which are already text
        contents.append(data)

    return contents



================================================
FILE: src/google/adk/flows/llm_flows/auto_flow.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of AutoFlow."""

from . import agent_transfer
from .single_flow import SingleFlow


class AutoFlow(SingleFlow):
  """AutoFlow is SingleFlow with agent transfer capability.

  Agent transfer is allowed in the following direction:

  1. from parent to sub-agent;
  2. from sub-agent to parent;
  3. from sub-agent to its peer agents;

  For peer-agent transfers, it's only enabled when all below conditions are met:

  - The parent agent is also of AutoFlow;
  - `disallow_transfer_to_peer` option of this agent is False (default).

  Depending on the target agent flow type, the transfer may be automatically
  reversed. The condition is as below:

  - If the flow type of the tranferee agent is also auto, transfee agent will
    remain as the active agent. The transfee agent will respond to the user's
    next message directly.
  - If the flow type of the transfere agent is not auto, the active agent will
    be reversed back to previous agent.

  TODO: allow user to config auto-reverse function.
  """

  def __init__(self):
    super().__init__()
    self.request_processors += [agent_transfer.request_processor]



================================================
FILE: src/google/adk/flows/llm_flows/base_llm_flow.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABC
import asyncio
import inspect
import logging
from typing import AsyncGenerator
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

from websockets.exceptions import ConnectionClosedOK

from ...agents.base_agent import BaseAgent
from ...agents.callback_context import CallbackContext
from ...agents.invocation_context import InvocationContext
from ...agents.live_request_queue import LiveRequestQueue
from ...agents.run_config import StreamingMode
from ...agents.transcription_entry import TranscriptionEntry
from ...events.event import Event
from ...models.base_llm_connection import BaseLlmConnection
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ...telemetry import trace_call_llm
from ...telemetry import trace_send_data
from ...telemetry import tracer
from ...tools.tool_context import ToolContext
from . import functions

if TYPE_CHECKING:
  from ...agents.llm_agent import LlmAgent
  from ...models.base_llm import BaseLlm
  from ._base_llm_processor import BaseLlmRequestProcessor
  from ._base_llm_processor import BaseLlmResponseProcessor

logger = logging.getLogger(__name__)


class BaseLlmFlow(ABC):
  """A basic flow that calls the LLM in a loop until a final response is generated.

  This flow ends when it transfer to another agent.
  """

  def __init__(self):
    self.request_processors: list[BaseLlmRequestProcessor] = []
    self.response_processors: list[BaseLlmResponseProcessor] = []

  async def run_live(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """Runs the flow using live api."""
    llm_request = LlmRequest()
    event_id = Event.new_id()

    # Preprocess before calling the LLM.
    async for event in self._preprocess_async(invocation_context, llm_request):
      yield event
    if invocation_context.end_invocation:
      return

    llm = self.__get_llm(invocation_context)
    logger.debug(
        'Establishing live connection for agent: %s with llm request: %s',
        invocation_context.agent.name,
        llm_request,
    )
    async with llm.connect(llm_request) as llm_connection:
      if llm_request.contents:
        # Sends the conversation history to the model.
        with tracer.start_as_current_span('send_data'):

          if invocation_context.transcription_cache:
            from . import audio_transcriber

            audio_transcriber = audio_transcriber.AudioTranscriber()
            contents = audio_transcriber.transcribe_file(invocation_context)
            logger.debug('Sending history to model: %s', contents)
            await llm_connection.send_history(contents)
            invocation_context.transcription_cache = None
            trace_send_data(invocation_context, event_id, contents)
          else:
            await llm_connection.send_history(llm_request.contents)
            trace_send_data(invocation_context, event_id, llm_request.contents)

      send_task = asyncio.create_task(
          self._send_to_model(llm_connection, invocation_context)
      )

      try:
        async for event in self._receive_from_model(
            llm_connection,
            event_id,
            invocation_context,
            llm_request,
        ):
          # Empty event means the queue is closed.
          if not event:
            break
          logger.debug('Receive new event: %s', event)
          yield event
          # send back the function response
          if event.get_function_responses():
            logger.debug('Sending back last function response event: %s', event)
            invocation_context.live_request_queue.send_content(event.content)
          if (
              event.content
              and event.content.parts
              and event.content.parts[0].function_response
              and event.content.parts[0].function_response.name
              == 'transfer_to_agent'
          ):
            await asyncio.sleep(1)
            # cancel the tasks that belongs to the closed connection.
            send_task.cancel()
            await llm_connection.close()
      finally:
        # Clean up
        if not send_task.done():
          send_task.cancel()
        try:
          await send_task
        except asyncio.CancelledError:
          pass

  async def _send_to_model(
      self,
      llm_connection: BaseLlmConnection,
      invocation_context: InvocationContext,
  ):
    """Sends data to model."""
    while True:
      live_request_queue = invocation_context.live_request_queue
      try:
        # Streamlit's execution model doesn't preemptively yield to the event
        # loop. Therefore, we must explicitly introduce timeouts to allow the
        # event loop to process events.
        # TODO: revert back(remove timeout) once we move off streamlit.
        live_request = await asyncio.wait_for(
            live_request_queue.get(), timeout=0.25
        )
        # duplicate the live_request to all the active streams
        logger.debug(
            'Sending live request %s to active streams: %s',
            live_request,
            invocation_context.active_streaming_tools,
        )
        if invocation_context.active_streaming_tools:
          for active_streaming_tool in (
              invocation_context.active_streaming_tools
          ).values():
            if active_streaming_tool.stream:
              active_streaming_tool.stream.send(live_request)
        await asyncio.sleep(0)
      except asyncio.TimeoutError:
        continue
      if live_request.close:
        await llm_connection.close()
        return
      if live_request.blob:
        # Cache audio data here for transcription
        if not invocation_context.transcription_cache:
          invocation_context.transcription_cache = []
        invocation_context.transcription_cache.append(
            TranscriptionEntry(role='user', data=live_request.blob)
        )
        await llm_connection.send_realtime(live_request.blob)
      if live_request.content:
        await llm_connection.send_content(live_request.content)

  async def _receive_from_model(
      self,
      llm_connection: BaseLlmConnection,
      event_id: str,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    """Receive data from model and process events using BaseLlmConnection."""
    def get_author(llm_response):
      """Get the author of the event.

      When the model returns transcription, the author is "user". Otherwise, the
      author is the agent.
      """
      if llm_response and llm_response.content and llm_response.content.role == "user":
        return "user"
      else:
        return invocation_context.agent.name

    assert invocation_context.live_request_queue
    try:
      while True:
        async for llm_response in llm_connection.receive():
          model_response_event = Event(
              id=Event.new_id(),
              invocation_id=invocation_context.invocation_id,
              author=get_author(llm_response),
          )
          async for event in self._postprocess_live(
              invocation_context,
              llm_request,
              llm_response,
              model_response_event,
          ):
            if (
                event.content
                and event.content.parts
                and event.content.parts[0].text
                and not event.partial
            ):
              if not invocation_context.transcription_cache:
                invocation_context.transcription_cache = []
              invocation_context.transcription_cache.append(
                  TranscriptionEntry(role='model', data=event.content)
              )
            yield event
        # Give opportunity for other tasks to run.
        await asyncio.sleep(0)
    except ConnectionClosedOK:
      pass

  async def run_async(
      self, invocation_context: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """Runs the flow."""
    while True:
      last_event = None
      async for event in self._run_one_step_async(invocation_context):
        last_event = event
        yield event
      if not last_event or last_event.is_final_response():
        break

  async def _run_one_step_async(
      self,
      invocation_context: InvocationContext,
  ) -> AsyncGenerator[Event, None]:
    """One step means one LLM call."""
    llm_request = LlmRequest()

    # Preprocess before calling the LLM.
    async for event in self._preprocess_async(invocation_context, llm_request):
      yield event
    if invocation_context.end_invocation:
      return

    # Calls the LLM.
    model_response_event = Event(
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        branch=invocation_context.branch,
    )
    async for llm_response in self._call_llm_async(
        invocation_context, llm_request, model_response_event
    ):
      # Postprocess after calling the LLM.
      async for event in self._postprocess_async(
          invocation_context, llm_request, llm_response, model_response_event
      ):
        # Use a new id for every event.
        event.id = Event.new_id()
        yield event

  async def _preprocess_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    # Runs processors.
    for processor in self.request_processors:
      async for event in processor.run_async(invocation_context, llm_request):
        yield event

    # Run processors for tools.
    for tool in agent.canonical_tools:
      tool_context = ToolContext(invocation_context)
      await tool.process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  async def _postprocess_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> AsyncGenerator[Event, None]:
    """Postprocess after calling the LLM.

    Args:
      invocation_context: The invocation context.
      llm_request: The original LLM request.
      llm_response: The LLM response from the LLM call.
      model_response_event: A mutable event for the LLM response.

    Yields:
      A generator of events.
    """

    # Runs processors.
    async for event in self._postprocess_run_processors_async(
        invocation_context, llm_response
    ):
      yield event

    # Skip the model response event if there is no content and no error code.
    # This is needed for the code executor to trigger another loop.
    if (
        not llm_response.content
        and not llm_response.error_code
        and not llm_response.interrupted
    ):
      return

    # Builds the event.
    model_response_event = self._finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )
    yield model_response_event

    # Handles function calls.
    if model_response_event.get_function_calls():
      async for event in self._postprocess_handle_function_calls_async(
          invocation_context, model_response_event, llm_request
      ):
        yield event

  async def _postprocess_live(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> AsyncGenerator[Event, None]:
    """Postprocess after calling the LLM asynchronously.

    Args:
      invocation_context: The invocation context.
      llm_request: The original LLM request.
      llm_response: The LLM response from the LLM call.
      model_response_event: A mutable event for the LLM response.

    Yields:
      A generator of events.
    """

    # Runs processors.
    async for event in self._postprocess_run_processors_async(
        invocation_context, llm_response
    ):
      yield event

    # Skip the model response event if there is no content and no error code.
    # This is needed for the code executor to trigger another loop.
    # But don't skip control events like turn_complete.
    if (
        not llm_response.content
        and not llm_response.error_code
        and not llm_response.interrupted
        and not llm_response.turn_complete
    ):
      return

    # Builds the event.
    model_response_event = self._finalize_model_response_event(
        llm_request, llm_response, model_response_event
    )
    yield model_response_event

    # Handles function calls.
    if model_response_event.get_function_calls():
      function_response_event = await functions.handle_function_calls_live(
          invocation_context, model_response_event, llm_request.tools_dict
      )
      yield function_response_event

      transfer_to_agent = function_response_event.actions.transfer_to_agent
      if transfer_to_agent:
        agent_to_run = self._get_agent_to_run(
            invocation_context, transfer_to_agent
        )
        async for item in agent_to_run.run_live(invocation_context):
          yield item

  async def _postprocess_run_processors_async(
      self, invocation_context: InvocationContext, llm_response: LlmResponse
  ) -> AsyncGenerator[Event, None]:
    for processor in self.response_processors:
      async for event in processor.run_async(invocation_context, llm_response):
        yield event

  async def _postprocess_handle_function_calls_async(
      self,
      invocation_context: InvocationContext,
      function_call_event: Event,
      llm_request: LlmRequest,
  ) -> AsyncGenerator[Event, None]:
    if function_response_event := await functions.handle_function_calls_async(
        invocation_context, function_call_event, llm_request.tools_dict
    ):
      auth_event = functions.generate_auth_event(
          invocation_context, function_response_event
      )
      if auth_event:
        yield auth_event

      yield function_response_event
      transfer_to_agent = function_response_event.actions.transfer_to_agent
      if transfer_to_agent:
        agent_to_run = self._get_agent_to_run(
            invocation_context, transfer_to_agent
        )
        async for event in agent_to_run.run_async(invocation_context):
          yield event

  def _get_agent_to_run(
      self, invocation_context: InvocationContext, transfer_to_agent
  ) -> BaseAgent:
    root_agent = invocation_context.agent.root_agent
    agent_to_run = root_agent.find_agent(transfer_to_agent)
    if not agent_to_run:
      raise ValueError(
          f'Agent {transfer_to_agent} not found in the agent tree.'
      )
    return agent_to_run

  async def _call_llm_async(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> AsyncGenerator[LlmResponse, None]:
    # Runs before_model_callback if it exists.
    if response := await self._handle_before_model_callback(
        invocation_context, llm_request, model_response_event
    ):
      yield response
      return

    # Calls the LLM.
    llm = self.__get_llm(invocation_context)
    with tracer.start_as_current_span('call_llm'):
      if invocation_context.run_config.support_cfc:
        invocation_context.live_request_queue = LiveRequestQueue()
        async for llm_response in self.run_live(invocation_context):
          # Runs after_model_callback if it exists.
          if altered_llm_response := await self._handle_after_model_callback(
              invocation_context, llm_response, model_response_event
          ):
            llm_response = altered_llm_response
          # only yield partial response in SSE streaming mode
          if (
              invocation_context.run_config.streaming_mode == StreamingMode.SSE
              or not llm_response.partial
          ):
            yield llm_response
          if llm_response.turn_complete:
            invocation_context.live_request_queue.close()
      else:
        # Check if we can make this llm call or not. If the current call pushes
        # the counter beyond the max set value, then the execution is stopped
        # right here, and exception is thrown.
        invocation_context.increment_llm_call_count()
        async for llm_response in llm.generate_content_async(
            llm_request,
            stream=invocation_context.run_config.streaming_mode
            == StreamingMode.SSE,
        ):
          trace_call_llm(
              invocation_context,
              model_response_event.id,
              llm_request,
              llm_response,
          )
          # Runs after_model_callback if it exists.
          if altered_llm_response := await self._handle_after_model_callback(
              invocation_context, llm_response, model_response_event
          ):
            llm_response = altered_llm_response

          yield llm_response

  async def _handle_before_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_request: LlmRequest,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    if not agent.canonical_before_model_callbacks:
      return

    callback_context = CallbackContext(
        invocation_context, event_actions=model_response_event.actions
    )

    for callback in agent.canonical_before_model_callbacks:
      before_model_callback_content = callback(
          callback_context=callback_context, llm_request=llm_request
      )
      if inspect.isawaitable(before_model_callback_content):
        before_model_callback_content = await before_model_callback_content
      if before_model_callback_content:
        return before_model_callback_content

  async def _handle_after_model_callback(
      self,
      invocation_context: InvocationContext,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Optional[LlmResponse]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    if not agent.canonical_after_model_callbacks:
      return

    callback_context = CallbackContext(
        invocation_context, event_actions=model_response_event.actions
    )

    for callback in agent.canonical_after_model_callbacks:
      after_model_callback_content = callback(
          callback_context=callback_context, llm_response=llm_response
      )
      if inspect.isawaitable(after_model_callback_content):
        after_model_callback_content = await after_model_callback_content
      if after_model_callback_content:
        return after_model_callback_content

  def _finalize_model_response_event(
      self,
      llm_request: LlmRequest,
      llm_response: LlmResponse,
      model_response_event: Event,
  ) -> Event:
    model_response_event = Event.model_validate({
        **model_response_event.model_dump(exclude_none=True),
        **llm_response.model_dump(exclude_none=True),
    })

    if model_response_event.content:
      function_calls = model_response_event.get_function_calls()
      if function_calls:
        functions.populate_client_function_call_id(model_response_event)
        model_response_event.long_running_tool_ids = (
            functions.get_long_running_function_calls(
                function_calls, llm_request.tools_dict
            )
        )

    return model_response_event

  def __get_llm(self, invocation_context: InvocationContext) -> BaseLlm:
    from ...agents.llm_agent import LlmAgent

    return cast(LlmAgent, invocation_context.agent).canonical_model



================================================
FILE: src/google/adk/flows/llm_flows/basic.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handles basic information to build the LLM request."""

from __future__ import annotations

from typing import AsyncGenerator
from typing import Generator

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ._base_llm_processor import BaseLlmRequestProcessor


class _BasicLlmRequestProcessor(BaseLlmRequestProcessor):

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    llm_request.model = (
        agent.canonical_model
        if isinstance(agent.canonical_model, str)
        else agent.canonical_model.model
    )
    llm_request.config = (
        agent.generate_content_config.model_copy(deep=True)
        if agent.generate_content_config
        else types.GenerateContentConfig()
    )
    if agent.output_schema:
      llm_request.set_output_schema(agent.output_schema)

    llm_request.live_connect_config.response_modalities = (
        invocation_context.run_config.response_modalities
    )
    llm_request.live_connect_config.speech_config = (
        invocation_context.run_config.speech_config
    )
    llm_request.live_connect_config.output_audio_transcription = (
        invocation_context.run_config.output_audio_transcription
    )
    llm_request.live_connect_config.input_audio_transcription = (
        invocation_context.run_config.input_audio_transcription
    )

    # TODO: handle tool append here, instead of in BaseTool.process_llm_request.

    return
    yield  # Generator requires yield statement in function body.


request_processor = _BasicLlmRequestProcessor()



================================================
FILE: src/google/adk/flows/llm_flows/contents.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
from typing import AsyncGenerator, Generator, Optional

from google.genai import types
from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ._base_llm_processor import BaseLlmRequestProcessor
from .functions import remove_client_function_call_id
from .functions import REQUEST_EUC_FUNCTION_CALL_NAME


class _ContentLlmRequestProcessor(BaseLlmRequestProcessor):
  """Builds the contents for the LLM request."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    if agent.include_contents != 'none':
      llm_request.contents = _get_contents(
          invocation_context.branch,
          invocation_context.session.events,
          agent.name,
      )

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _ContentLlmRequestProcessor()


def _rearrange_events_for_async_function_responses_in_history(
    events: list[Event],
) -> list[Event]:
  """Rearrange the async function_response events in the history."""

  function_call_id_to_response_events_index: dict[str, list[Event]] = {}
  for i, event in enumerate(events):
    function_responses = event.get_function_responses()
    if function_responses:
      for function_response in function_responses:
        function_call_id = function_response.id
        function_call_id_to_response_events_index[function_call_id] = i

  result_events: list[Event] = []
  for event in events:
    if event.get_function_responses():
      # function_response should be handled together with function_call below.
      continue
    elif event.get_function_calls():

      function_response_events_indices = set()
      for function_call in event.get_function_calls():
        function_call_id = function_call.id
        if function_call_id in function_call_id_to_response_events_index:
          function_response_events_indices.add(
              function_call_id_to_response_events_index[function_call_id]
          )
      result_events.append(event)
      if not function_response_events_indices:
        continue
      if len(function_response_events_indices) == 1:
        result_events.append(
            events[next(iter(function_response_events_indices))]
        )
      else:  # Merge all async function_response as one response event
        result_events.append(
            _merge_function_response_events(
                [events[i] for i in sorted(function_response_events_indices)]
            )
        )
      continue
    else:
      result_events.append(event)

  return result_events


def _rearrange_events_for_latest_function_response(
    events: list[Event],
) -> list[Event]:
  """Rearrange the events for the latest function_response.

  If the latest function_response is for an async function_call, all events
  between the initial function_call and the latest function_response will be
  removed.

  Args:
    events: A list of events.

  Returns:
    A list of events with the latest function_response rearranged.
  """
  if not events:
    return events

  function_responses = events[-1].get_function_responses()
  if not function_responses:
    # No need to process, since the latest event is not fuction_response.
    return events

  function_responses_ids = set()
  for function_response in function_responses:
    function_responses_ids.add(function_response.id)

  function_calls = events[-2].get_function_calls()

  if function_calls:
    for function_call in function_calls:
      # The latest function_response is already matched
      if function_call.id in function_responses_ids:
        return events

  function_call_event_idx = -1
  # look for corresponding function call event reversely
  for idx in range(len(events) - 2, -1, -1):
    event = events[idx]
    function_calls = event.get_function_calls()
    if function_calls:
      for function_call in function_calls:
        if function_call.id in function_responses_ids:
          function_call_event_idx = idx
          break
        if function_call_event_idx != -1:
          # in case the last response event only have part of the responses
          # for the function calls in the function call event
          for function_call in function_calls:
            function_responses_ids.add(function_call.id)
          break

  if function_call_event_idx == -1:
    raise ValueError(
        'No function call event found for function responses ids:'
        f' {function_responses_ids}'
    )

  # collect all function response between last function response event
  # and function call event

  function_response_events: list[Event] = []
  for idx in range(function_call_event_idx + 1, len(events) - 1):
    event = events[idx]
    function_responses = event.get_function_responses()
    if (
        function_responses
        and function_responses[0].id in function_responses_ids
    ):
      function_response_events.append(event)
  function_response_events.append(events[-1])

  result_events = events[: function_call_event_idx + 1]
  result_events.append(
      _merge_function_response_events(function_response_events)
  )

  return result_events


def _get_contents(
    current_branch: Optional[str], events: list[Event], agent_name: str = ''
) -> list[types.Content]:
  """Get the contents for the LLM request.

  Args:
    current_branch: The current branch of the agent.
    events: A list of events.
    agent_name: The name of the agent.

  Returns:
    A list of contents.
  """
  filtered_events = []
  # Parse the events, leaving the contents and the function calls and
  # responses from the current agent.
  for event in events:
    if (
        not event.content
        or not event.content.role
        or not event.content.parts
        or event.content.parts[0].text == ''
    ):
      # Skip events without content, or generated neither by user nor by model
      # or has empty text.
      # E.g. events purely for mutating session states.
      continue
    if not _is_event_belongs_to_branch(current_branch, event):
      # Skip events not belong to current branch.
      continue
    if _is_auth_event(event):
      # skip auth event
      continue
    filtered_events.append(
        _convert_foreign_event(event)
        if _is_other_agent_reply(agent_name, event)
        else event
    )

  result_events = _rearrange_events_for_latest_function_response(
      filtered_events
  )
  result_events = _rearrange_events_for_async_function_responses_in_history(
      result_events
  )
  contents = []
  for event in result_events:
    content = copy.deepcopy(event.content)
    remove_client_function_call_id(content)
    contents.append(content)
  return contents


def _is_other_agent_reply(current_agent_name: str, event: Event) -> bool:
  """Whether the event is a reply from another agent."""
  return bool(
      current_agent_name
      and event.author != current_agent_name
      and event.author != 'user'
  )


def _convert_foreign_event(event: Event) -> Event:
  """Converts an event authored by another agent as a user-content event.

  This is to provide another agent's output as context to the current agent, so
  that current agent can continue to respond, such as summarizing previous
  agent's reply, etc.

  Args:
    event: The event to convert.

  Returns:
    The converted event.

  """
  if not event.content or not event.content.parts:
    return event

  content = types.Content()
  content.role = 'user'
  content.parts = [types.Part(text='For context:')]
  for part in event.content.parts:
    if part.text:
      content.parts.append(
          types.Part(text=f'[{event.author}] said: {part.text}')
      )
    elif part.function_call:
      content.parts.append(
          types.Part(
              text=(
                  f'[{event.author}] called tool `{part.function_call.name}`'
                  f' with parameters: {part.function_call.args}'
              )
          )
      )
    elif part.function_response:
      # Otherwise, create a new text part.
      content.parts.append(
          types.Part(
              text=(
                  f'[{event.author}] `{part.function_response.name}` tool'
                  f' returned result: {part.function_response.response}'
              )
          )
      )
    # Fallback to the original part for non-text and non-functionCall parts.
    else:
      content.parts.append(part)

  return Event(
      timestamp=event.timestamp,
      author='user',
      content=content,
      branch=event.branch,
  )


def _merge_function_response_events(
    function_response_events: list[Event],
) -> Event:
  """Merges a list of function_response events into one event.

  The key goal is to ensure:
  1. function_call and function_response are always of the same number.
  2. The function_call and function_response are consecutively in the content.

  Args:
    function_response_events: A list of function_response events.
      NOTE: function_response_events must fulfill these requirements: 1. The
        list is in increasing order of timestamp; 2. the first event is the
        initial function_response event; 3. all later events should contain at
        least one function_response part that related to the function_call
        event. (Note, 3. may not be true when aync function return some
        intermediate response, there could also be some intermediate model
        response event without any function_response and such event will be
        ignored.)
      Caveat: This implementation doesn't support when a parallel function_call
        event contains async function_call of the same name.

  Returns:
    A merged event, that is
      1. All later function_response will replace function_response part in
          the initial function_response event.
      2. All non-function_response parts will be appended to the part list of
          the initial function_response event.
  """
  if not function_response_events:
    raise ValueError('At least one function_response event is required.')

  merged_event = function_response_events[0].model_copy(deep=True)
  parts_in_merged_event: list[types.Part] = merged_event.content.parts  # type: ignore

  if not parts_in_merged_event:
    raise ValueError('There should be at least one function_response part.')

  part_indices_in_merged_event: dict[str, int] = {}
  for idx, part in enumerate(parts_in_merged_event):
    if part.function_response:
      function_call_id: str = part.function_response.id  # type: ignore
      part_indices_in_merged_event[function_call_id] = idx

  for event in function_response_events[1:]:
    if not event.content.parts:
      raise ValueError('There should be at least one function_response part.')

    for part in event.content.parts:
      if part.function_response:
        function_call_id: str = part.function_response.id  # type: ignore
        if function_call_id in part_indices_in_merged_event:
          parts_in_merged_event[
              part_indices_in_merged_event[function_call_id]
          ] = part
        else:
          parts_in_merged_event.append(part)
          part_indices_in_merged_event[function_call_id] = (
              len(parts_in_merged_event) - 1
          )

      else:
        parts_in_merged_event.append(part)

  return merged_event


def _is_event_belongs_to_branch(
    invocation_branch: Optional[str], event: Event
) -> bool:
  """Event belongs to a branch, when event.branch is prefix of the invocation branch."""
  if not invocation_branch or not event.branch:
    return True
  return invocation_branch.startswith(event.branch)


def _is_auth_event(event: Event) -> bool:
  if not event.content.parts:
    return False
  for part in event.content.parts:
    if (
        part.function_call
        and part.function_call.name == REQUEST_EUC_FUNCTION_CALL_NAME
    ):
      return True
    if (
        part.function_response
        and part.function_response.name == REQUEST_EUC_FUNCTION_CALL_NAME
    ):
      return True
  return False



================================================
FILE: src/google/adk/flows/llm_flows/functions.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handles function callings for LLM flow."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Optional
import uuid

from google.genai import types

from ...agents.active_streaming_tool import ActiveStreamingTool
from ...agents.invocation_context import InvocationContext
from ...auth.auth_tool import AuthToolArguments
from ...events.event import Event
from ...events.event_actions import EventActions
from ...telemetry import trace_tool_call
from ...telemetry import trace_tool_response
from ...telemetry import tracer
from ...tools.base_tool import BaseTool
from ...tools.tool_context import ToolContext

AF_FUNCTION_CALL_ID_PREFIX = 'adk-'
REQUEST_EUC_FUNCTION_CALL_NAME = 'adk_request_credential'

logger = logging.getLogger(__name__)


def generate_client_function_call_id() -> str:
  return f'{AF_FUNCTION_CALL_ID_PREFIX}{uuid.uuid4()}'


def populate_client_function_call_id(model_response_event: Event) -> None:
  if not model_response_event.get_function_calls():
    return
  for function_call in model_response_event.get_function_calls():
    if not function_call.id:
      function_call.id = generate_client_function_call_id()


def remove_client_function_call_id(content: types.Content) -> None:
  if content and content.parts:
    for part in content.parts:
      if (
          part.function_call
          and part.function_call.id
          and part.function_call.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      ):
        part.function_call.id = None
      if (
          part.function_response
          and part.function_response.id
          and part.function_response.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      ):
        part.function_response.id = None


def get_long_running_function_calls(
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
) -> set[str]:
  long_running_tool_ids = set()
  for function_call in function_calls:
    if (
        function_call.name in tools_dict
        and tools_dict[function_call.name].is_long_running
    ):
      long_running_tool_ids.add(function_call.id)

  return long_running_tool_ids


def generate_auth_event(
    invocation_context: InvocationContext,
    function_response_event: Event,
) -> Optional[Event]:
  if not function_response_event.actions.requested_auth_configs:
    return None
  parts = []
  long_running_tool_ids = set()
  for (
      function_call_id,
      auth_config,
  ) in function_response_event.actions.requested_auth_configs.items():

    request_euc_function_call = types.FunctionCall(
        name=REQUEST_EUC_FUNCTION_CALL_NAME,
        args=AuthToolArguments(
            function_call_id=function_call_id,
            auth_config=auth_config,
        ).model_dump(exclude_none=True),
    )
    request_euc_function_call.id = generate_client_function_call_id()
    long_running_tool_ids.add(request_euc_function_call.id)
    parts.append(types.Part(function_call=request_euc_function_call))

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      branch=invocation_context.branch,
      content=types.Content(
          parts=parts, role=function_response_event.content.role
      ),
      long_running_tool_ids=long_running_tool_ids,
  )


async def handle_function_calls_async(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
) -> Optional[Event]:
  """Calls the functions and returns the function response event."""
  from ...agents.llm_agent import LlmAgent

  agent = invocation_context.agent
  if not isinstance(agent, LlmAgent):
    return

  function_calls = function_call_event.get_function_calls()

  function_response_events: list[Event] = []
  for function_call in function_calls:
    if filters and function_call.id not in filters:
      continue
    tool, tool_context = _get_tool_and_context(
        invocation_context,
        function_call_event,
        function_call,
        tools_dict,
    )
    # do not use "args" as the variable name, because it is a reserved keyword
    # in python debugger.
    function_args = function_call.args or {}
    function_response: Optional[dict] = None

    # before_tool_callback (sync or async)
    if agent.before_tool_callback:
      function_response = agent.before_tool_callback(
          tool=tool, args=function_args, tool_context=tool_context
      )
      if inspect.isawaitable(function_response):
        function_response = await function_response

    if not function_response:
      function_response = await __call_tool_async(
          tool, args=function_args, tool_context=tool_context
      )

    # after_tool_callback (sync or async)
    if agent.after_tool_callback:
      altered_function_response = agent.after_tool_callback(
          tool=tool,
          args=function_args,
          tool_context=tool_context,
          tool_response=function_response,
      )
      if inspect.isawaitable(altered_function_response):
        altered_function_response = await altered_function_response
      if altered_function_response is not None:
        function_response = altered_function_response

    if tool.is_long_running:
      # Allow long running function to return None to not provide function response.
      if not function_response:
        continue

    # Builds the function response event.
    function_response_event = __build_response_event(
        tool, function_response, tool_context, invocation_context
    )
    function_response_events.append(function_response_event)

  if not function_response_events:
    return None
  merged_event = merge_parallel_function_response_events(
      function_response_events
  )
  if len(function_response_events) > 1:
    # this is needed for debug traces of parallel calls
    # individual response with tool.name is traced in __build_response_event
    # (we drop tool.name from span name here as this is merged event)
    with tracer.start_as_current_span('tool_response'):
      trace_tool_response(
          invocation_context=invocation_context,
          event_id=merged_event.id,
          function_response_event=merged_event,
      )
  return merged_event


async def handle_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
) -> Event:
  """Calls the functions and returns the function response event."""
  from ...agents.llm_agent import LlmAgent

  agent = cast(LlmAgent, invocation_context.agent)
  function_calls = function_call_event.get_function_calls()

  function_response_events: list[Event] = []
  for function_call in function_calls:
    tool, tool_context = _get_tool_and_context(
        invocation_context, function_call_event, function_call, tools_dict
    )
    # do not use "args" as the variable name, because it is a reserved keyword
    # in python debugger.
    function_args = function_call.args or {}
    function_response = None
    # # Calls the tool if before_tool_callback does not exist or returns None.
    # if agent.before_tool_callback:
    #   function_response = agent.before_tool_callback(
    #       tool, function_args, tool_context
    #   )
    if agent.before_tool_callback:
      function_response = agent.before_tool_callback(
          tool=tool, args=function_args, tool_context=tool_context
      )
      if inspect.isawaitable(function_response):
        function_response = await function_response

    if not function_response:
      function_response = await _process_function_live_helper(
          tool, tool_context, function_call, function_args, invocation_context
      )

    # Calls after_tool_callback if it exists.
    # if agent.after_tool_callback:
    #   new_response = agent.after_tool_callback(
    #       tool,
    #       function_args,
    #       tool_context,
    #       function_response,
    #   )
    #   if new_response:
    #     function_response = new_response
    if agent.after_tool_callback:
      altered_function_response = agent.after_tool_callback(
          tool=tool,
          args=function_args,
          tool_context=tool_context,
          tool_response=function_response,
      )
      if inspect.isawaitable(altered_function_response):
        altered_function_response = await altered_function_response
      if altered_function_response is not None:
        function_response = altered_function_response

    if tool.is_long_running:
      # Allow async function to return None to not provide function response.
      if not function_response:
        continue

    # Builds the function response event.
    function_response_event = __build_response_event(
        tool, function_response, tool_context, invocation_context
    )
    function_response_events.append(function_response_event)

  if not function_response_events:
    return None
  merged_event = merge_parallel_function_response_events(
      function_response_events
  )
  return merged_event


async def _process_function_live_helper(
    tool, tool_context, function_call, function_args, invocation_context
):
  function_response = None
  # Check if this is a stop_streaming function call
  if (
      function_call.name == 'stop_streaming'
      and 'function_name' in function_args
  ):
    function_name = function_args['function_name']
    active_tasks = invocation_context.active_streaming_tools
    if (
        function_name in active_tasks
        and active_tasks[function_name].task
        and not active_tasks[function_name].task.done()
    ):
      task = active_tasks[function_name].task
      task.cancel()
      try:
        # Wait for the task to be cancelled
        await asyncio.wait_for(task, timeout=1.0)
      except (asyncio.CancelledError, asyncio.TimeoutError):
        # Log the specific condition
        if task.cancelled():
          logging.info(f'Task {function_name} was cancelled successfully')
        elif task.done():
          logging.info(f'Task {function_name} completed during cancellation')
        else:
          logging.warning(
              f'Task {function_name} might still be running after'
              ' cancellation timeout'
          )
          function_response = {
              'status': f'The task is not cancelled yet for {function_name}.'
          }
      if not function_response:
        # Clean up the reference
        active_tasks[function_name].task = None

        function_response = {
            'status': f'Successfully stopped streaming function {function_name}'
        }
    else:
      function_response = {
          'status': f'No active streaming function named {function_name} found'
      }
  elif hasattr(tool, 'func') and inspect.isasyncgenfunction(tool.func):
    # for streaming tool use case
    # we require the function to be a async generator function
    async def run_tool_and_update_queue(tool, function_args, tool_context):
      try:
        async for result in __call_tool_live(
            tool=tool,
            args=function_args,
            tool_context=tool_context,
            invocation_context=invocation_context,
        ):
          updated_content = types.Content(
              role='user',
              parts=[
                  types.Part.from_text(
                      text=f'Function {tool.name} returned: {result}'
                  )
              ],
          )
          invocation_context.live_request_queue.send_content(updated_content)
      except asyncio.CancelledError:
        raise  # Re-raise to properly propagate the cancellation

    task = asyncio.create_task(
        run_tool_and_update_queue(tool, function_args, tool_context)
    )
    if invocation_context.active_streaming_tools is None:
      invocation_context.active_streaming_tools = {}
    if tool.name in invocation_context.active_streaming_tools:
      invocation_context.active_streaming_tools[tool.name].task = task
    else:
      invocation_context.active_streaming_tools[tool.name] = (
          ActiveStreamingTool(task=task)
      )
    # Immediately return a pending response.
    # This is required by current live model.
    function_response = {
        'status': (
            'The function is running asynchronously and the results are'
            ' pending.'
        )
    }
  else:
    function_response = await __call_tool_async(
        tool, args=function_args, tool_context=tool_context
    )
  return function_response


def _get_tool_and_context(
    invocation_context: InvocationContext,
    function_call_event: Event,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
):
  if function_call.name not in tools_dict:
    raise ValueError(
        f'Function {function_call.name} is not found in the tools_dict.'
    )

  tool_context = ToolContext(
      invocation_context=invocation_context,
      function_call_id=function_call.id,
  )

  tool = tools_dict[function_call.name]

  return (tool, tool_context)


async def __call_tool_live(
    tool: BaseTool,
    args: dict[str, object],
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> AsyncGenerator[Event, None]:
  """Calls the tool asynchronously (awaiting the coroutine)."""
  with tracer.start_as_current_span(f'tool_call [{tool.name}]'):
    trace_tool_call(args=args)
    async for item in tool._call_live(
        args=args,
        tool_context=tool_context,
        invocation_context=invocation_context,
    ):
      yield item


async def __call_tool_async(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Any:
  """Calls the tool."""
  with tracer.start_as_current_span(f'tool_call [{tool.name}]'):
    trace_tool_call(args=args)
    return await tool.run_async(args=args, tool_context=tool_context)


def __build_response_event(
    tool: BaseTool,
    function_result: dict[str, object],
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> Event:
  with tracer.start_as_current_span(f'tool_response [{tool.name}]'):
    # Specs requires the result to be a dict.
    if not isinstance(function_result, dict):
      function_result = {'result': function_result}

    part_function_response = types.Part.from_function_response(
        name=tool.name, response=function_result
    )
    part_function_response.function_response.id = tool_context.function_call_id

    content = types.Content(
        role='user',
        parts=[part_function_response],
    )

    function_response_event = Event(
        invocation_id=invocation_context.invocation_id,
        author=invocation_context.agent.name,
        content=content,
        actions=tool_context.actions,
        branch=invocation_context.branch,
    )

    trace_tool_response(
        invocation_context=invocation_context,
        event_id=function_response_event.id,
        function_response_event=function_response_event,
    )
    return function_response_event


def merge_parallel_function_response_events(
    function_response_events: list['Event'],
) -> 'Event':
  if not function_response_events:
    raise ValueError('No function response events provided.')

  if len(function_response_events) == 1:
    return function_response_events[0]
  merged_parts = []
  for event in function_response_events:
    if event.content:
      for part in event.content.parts or []:
        merged_parts.append(part)

  # Use the first event as the "base" for common attributes
  base_event = function_response_events[0]

  # Merge actions from all events

  merged_actions = EventActions()
  merged_requested_auth_configs = {}
  for event in function_response_events:
    merged_requested_auth_configs.update(event.actions.requested_auth_configs)
    merged_actions = merged_actions.model_copy(
        update=event.actions.model_dump()
    )
  merged_actions.requested_auth_configs = merged_requested_auth_configs
  # Create the new merged event
  merged_event = Event(
      invocation_id=Event.new_id(),
      author=base_event.author,
      branch=base_event.branch,
      content=types.Content(role='user', parts=merged_parts),
      actions=merged_actions,  # Optionally merge actions if required
  )

  # Use the base_event as the timestamp
  merged_event.timestamp = base_event.timestamp
  return merged_event



================================================
FILE: src/google/adk/flows/llm_flows/identity.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gives the agent identity from the framework."""

from __future__ import annotations

from typing import AsyncGenerator

from typing_extensions import override

from ...agents.invocation_context import InvocationContext
from ...events.event import Event
from ...models.llm_request import LlmRequest
from ._base_llm_processor import BaseLlmRequestProcessor


class _IdentityLlmRequestProcessor(BaseLlmRequestProcessor):
  """Gives the agent identity from the framework."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    agent = invocation_context.agent
    si = [f'You are an agent. Your internal name is "{agent.name}".']
    if agent.description:
      si.append(f' The description about you is "{agent.description}"')
    llm_request.append_instructions(si)

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _IdentityLlmRequestProcessor()



================================================
FILE: src/google/adk/flows/llm_flows/instructions.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handles instructions and global instructions for LLM flow."""

from __future__ import annotations

import re
from typing import AsyncGenerator
from typing import Generator
from typing import TYPE_CHECKING

from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...events.event import Event
from ...sessions.state import State
from ._base_llm_processor import BaseLlmRequestProcessor

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...models.llm_request import LlmRequest


class _InstructionsLlmRequestProcessor(BaseLlmRequestProcessor):
  """Handles instructions and global instructions for LLM flow."""

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    from ...agents.base_agent import BaseAgent
    from ...agents.llm_agent import LlmAgent

    agent = invocation_context.agent
    if not isinstance(agent, LlmAgent):
      return

    root_agent: BaseAgent = agent.root_agent

    # Appends global instructions if set.
    if (
        isinstance(root_agent, LlmAgent) and root_agent.global_instruction
    ):  # not empty str
      raw_si = root_agent.canonical_global_instruction(
          ReadonlyContext(invocation_context)
      )
      si = await _populate_values(raw_si, invocation_context)
      llm_request.append_instructions([si])

    # Appends agent instructions if set.
    if agent.instruction:  # not empty str
      raw_si = agent.canonical_instruction(ReadonlyContext(invocation_context))
      si = await _populate_values(raw_si, invocation_context)
      llm_request.append_instructions([si])

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _InstructionsLlmRequestProcessor()


async def _populate_values(
    instruction_template: str,
    context: InvocationContext,
) -> str:
  """Populates values in the instruction template, e.g. state, artifact, etc."""

  async def _async_sub(pattern, repl_async_fn, string) -> str:
    result = []
    last_end = 0
    for match in re.finditer(pattern, string):
      result.append(string[last_end : match.start()])
      replacement = await repl_async_fn(match)
      result.append(replacement)
      last_end = match.end()
    result.append(string[last_end:])
    return ''.join(result)

  async def _replace_match(match) -> str:
    var_name = match.group().lstrip('{').rstrip('}').strip()
    optional = False
    if var_name.endswith('?'):
      optional = True
      var_name = var_name.removesuffix('?')
    if var_name.startswith('artifact.'):
      var_name = var_name.removeprefix('artifact.')
      if context.artifact_service is None:
        raise ValueError('Artifact service is not initialized.')
      artifact = await context.artifact_service.load_artifact(
          app_name=context.session.app_name,
          user_id=context.session.user_id,
          session_id=context.session.id,
          filename=var_name,
      )
      if not var_name:
        raise KeyError(f'Artifact {var_name} not found.')
      return str(artifact)
    else:
      if not _is_valid_state_name(var_name):
        return match.group()
      if var_name in context.session.state:
        return str(context.session.state[var_name])
      else:
        if optional:
          return ''
        else:
          raise KeyError(f'Context variable not found: `{var_name}`.')

  return await _async_sub(r'{+[^{}]*}+', _replace_match, instruction_template)


def _is_valid_state_name(var_name):
  """Checks if the variable name is a valid state name.

  Valid state is either:
    - Valid identifier
    - <Valid prefix>:<Valid identifier>
  All the others will just return as it is.

  Args:
    var_name: The variable name to check.

  Returns:
    True if the variable name is a valid state name, False otherwise.
  """
  parts = var_name.split(':')
  if len(parts) == 1:
    return var_name.isidentifier()

  if len(parts) == 2:
    prefixes = [State.APP_PREFIX, State.USER_PREFIX, State.TEMP_PREFIX]
    if (parts[0] + ':') in prefixes:
      return parts[1].isidentifier()
  return False



================================================
FILE: src/google/adk/flows/llm_flows/single_flow.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of single flow."""

import logging

from ...auth import auth_preprocessor
from . import _code_execution
from . import _nl_planning
from . import basic
from . import contents
from . import identity
from . import instructions
from .base_llm_flow import BaseLlmFlow

logger = logging.getLogger(__name__)


class SingleFlow(BaseLlmFlow):
  """SingleFlow is the LLM flows that handles tools calls.

  A single flow only consider an agent itself and tools.
  No sub-agents are allowed for single flow.
  """

  def __init__(self):
    super().__init__()
    self.request_processors += [
        basic.request_processor,
        auth_preprocessor.request_processor,
        instructions.request_processor,
        identity.request_processor,
        contents.request_processor,
        # Some implementations of NL Planning mark planning contents as thoughts
        # in the post processor. Since these need to be unmarked, NL Planning
        # should be after contents.
        _nl_planning.request_processor,
        # Code execution should be after the contents as it mutates the contents
        # to optimize data files.
        _code_execution.request_processor,
    ]
    self.response_processors += [
        _nl_planning.response_processor,
        _code_execution.response_processor,
    ]



================================================
FILE: src/google/adk/memory/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

from .base_memory_service import BaseMemoryService
from .in_memory_memory_service import InMemoryMemoryService

logger = logging.getLogger(__name__)

__all__ = [
    'BaseMemoryService',
    'InMemoryMemoryService',
]

try:
  from .vertex_ai_rag_memory_service import VertexAiRagMemoryService

  __all__.append('VertexAiRagMemoryService')
except ImportError:
  logger.debug(
      'The Vertex sdk is not installed. If you want to use the'
      ' VertexAiRagMemoryService please install it. If not, you can ignore this'
      ' warning.'
  )



================================================
FILE: src/google/adk/memory/base_memory_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc

from pydantic import BaseModel
from pydantic import Field

from ..events.event import Event
from ..sessions.session import Session


class MemoryResult(BaseModel):
  """Represents a single memory retrieval result.

  Attributes:
      session_id: The session id associated with the memory.
      events: A list of events in the session.
  """

  session_id: str
  events: list[Event]


class SearchMemoryResponse(BaseModel):
  """Represents the response from a memory search.

  Attributes:
      memories: A list of memory results matching the search query.
  """

  memories: list[MemoryResult] = Field(default_factory=list)


class BaseMemoryService(abc.ABC):
  """Base class for memory services.

  The service provides functionalities to ingest sessions into memory so that
  the memory can be used for user queries.
  """

  @abc.abstractmethod
  async def add_session_to_memory(self, session: Session):
    """Adds a session to the memory service.

    A session may be added multiple times during its lifetime.

    Args:
        session: The session to add.
    """

  @abc.abstractmethod
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches for sessions that match the query.

    Args:
        app_name: The name of the application.
        user_id: The id of the user.
        query: The query to search for.

    Returns:
        A SearchMemoryResponse containing the matching memories.
    """



================================================
FILE: src/google/adk/memory/in_memory_memory_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..events.event import Event
from ..sessions.session import Session
from .base_memory_service import BaseMemoryService
from .base_memory_service import MemoryResult
from .base_memory_service import SearchMemoryResponse


class InMemoryMemoryService(BaseMemoryService):
  """An in-memory memory service for prototyping purpose only.

  Uses keyword matching instead of semantic search.
  """

  def __init__(self):
    self.session_events: dict[str, list[Event]] = {}
    """keys are app_name/user_id/session_id"""

  async def add_session_to_memory(self, session: Session):
    key = f'{session.app_name}/{session.user_id}/{session.id}'
    self.session_events[key] = [
        event for event in session.events if event.content
    ]

  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Prototyping purpose only."""
    keywords = set(query.lower().split())
    response = SearchMemoryResponse()
    for key, events in self.session_events.items():
      if not key.startswith(f'{app_name}/{user_id}/'):
        continue
      matched_events = []
      for event in events:
        if not event.content or not event.content.parts:
          continue
        parts = event.content.parts
        text = '\n'.join([part.text for part in parts if part.text]).lower()
        for keyword in keywords:
          if keyword in text:
            matched_events.append(event)
            break
      if matched_events:
        session_id = key.split('/')[-1]
        response.memories.append(
            MemoryResult(session_id=session_id, events=matched_events)
        )
    return response



================================================
FILE: src/google/adk/memory/vertex_ai_rag_memory_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
import json
import os
import tempfile

from google.genai import types
from typing_extensions import override
from vertexai.preview import rag

from ..events.event import Event
from ..sessions.session import Session
from .base_memory_service import BaseMemoryService
from .base_memory_service import MemoryResult
from .base_memory_service import SearchMemoryResponse


class VertexAiRagMemoryService(BaseMemoryService):
  """A memory service that uses Vertex AI RAG for storage and retrieval."""

  def __init__(
      self,
      rag_corpus: str = None,
      similarity_top_k: int = None,
      vector_distance_threshold: float = 10,
  ):
    """Initializes a VertexAiRagMemoryService.

    Args:
        rag_corpus: The name of the Vertex AI RAG corpus to use. Format:
          ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}``
          or ``{rag_corpus_id}``
        similarity_top_k: The number of contexts to retrieve.
        vector_distance_threshold: Only returns contexts with vector distance
          smaller than the threshold..
    """
    self.vertex_rag_store = types.VertexRagStore(
        rag_resources=[rag.RagResource(rag_corpus=rag_corpus)],
        similarity_top_k=similarity_top_k,
        vector_distance_threshold=vector_distance_threshold,
    )

  @override
  async def add_session_to_memory(self, session: Session):
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt"
    ) as temp_file:

      output_lines = []
      for event in session.events:
        if not event.content or not event.content.parts:
          continue
        text_parts = [
            part.text.replace("\n", " ")
            for part in event.content.parts
            if part.text
        ]
        if text_parts:
          output_lines.append(
              json.dumps({
                  "author": event.author,
                  "timestamp": event.timestamp,
                  "text": ".".join(text_parts),
              })
          )
      output_string = "\n".join(output_lines)
      temp_file.write(output_string)
      temp_file_path = temp_file.name
    for rag_resource in self.vertex_rag_store.rag_resources:
      rag.upload_file(
          corpus_name=rag_resource.rag_corpus,
          path=temp_file_path,
          # this is the temp workaround as upload file does not support
          # adding metadata, thus use display_name to store the session info.
          display_name=f"{session.app_name}.{session.user_id}.{session.id}",
      )

    os.remove(temp_file_path)

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches for sessions that match the query using rag.retrieval_query."""
    response = rag.retrieval_query(
        text=query,
        rag_resources=self.vertex_rag_store.rag_resources,
        rag_corpora=self.vertex_rag_store.rag_corpora,
        similarity_top_k=self.vertex_rag_store.similarity_top_k,
        vector_distance_threshold=self.vertex_rag_store.vector_distance_threshold,
    )

    memory_results = []
    session_events_map = OrderedDict()
    for context in response.contexts.contexts:
      # filter out context that is not related
      # TODO: Add server side filtering by app_name and user_id.
      # if not context.source_display_name.startswith(f"{app_name}.{user_id}."):
      #   continue
      session_id = context.source_display_name.split(".")[-1]
      events = []
      if context.text:
        lines = context.text.split("\n")

        for line in lines:
          line = line.strip()
          if not line:
            continue

          try:
            # Try to parse as JSON
            event_data = json.loads(line)

            author = event_data.get("author", "")
            timestamp = float(event_data.get("timestamp", 0))
            text = event_data.get("text", "")

            content = types.Content(parts=[types.Part(text=text)])
            event = Event(author=author, timestamp=timestamp, content=content)
            events.append(event)
          except json.JSONDecodeError:
            # Not valid JSON, skip this line
            continue

      if session_id in session_events_map:
        session_events_map[session_id].append(events)
      else:
        session_events_map[session_id] = [events]

    # Remove overlap and combine events from the same session.
    for session_id, event_lists in session_events_map.items():
      for events in _merge_event_lists(event_lists):
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        memory_results.append(
            MemoryResult(session_id=session_id, events=sorted_events)
        )
    return SearchMemoryResponse(memories=memory_results)


def _merge_event_lists(event_lists: list[list[Event]]) -> list[list[Event]]:
  """Merge event lists that have overlapping timestamps."""
  merged = []
  while event_lists:
    current = event_lists.pop(0)
    current_ts = {event.timestamp for event in current}
    merge_found = True

    # Keep merging until no new overlap is found.
    while merge_found:
      merge_found = False
      remaining = []
      for other in event_lists:
        other_ts = {event.timestamp for event in other}
        # Overlap exists, so we merge and use the merged list to check again
        if current_ts & other_ts:
          new_events = [e for e in other if e.timestamp not in current_ts]
          current.extend(new_events)
          current_ts.update(e.timestamp for e in new_events)
          merge_found = True
        else:
          remaining.append(other)
      event_lists = remaining
    merged.append(current)
  return merged



================================================
FILE: src/google/adk/models/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the interface to support a model."""

from .base_llm import BaseLlm
from .google_llm import Gemini
from .llm_request import LlmRequest
from .llm_response import LlmResponse
from .registry import LLMRegistry

__all__ = [
    'BaseLlm',
    'Gemini',
    'LLMRegistry',
]


for regex in Gemini.supported_models():
  LLMRegistry.register(Gemini)



================================================
FILE: src/google/adk/models/anthropic_llm.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic integration for Claude models."""

from __future__ import annotations

from functools import cached_property
import logging
import os
from typing import Any
from typing import AsyncGenerator
from typing import Generator
from typing import Iterable
from typing import Literal
from typing import Optional, Union
from typing import TYPE_CHECKING

from anthropic import AnthropicVertex
from anthropic import NOT_GIVEN
from anthropic import types as anthropic_types
from google.genai import types
from pydantic import BaseModel
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .llm_request import LlmRequest

__all__ = ["Claude"]

logger = logging.getLogger(__name__)

MAX_TOKEN = 1024


class ClaudeRequest(BaseModel):
  system_instruction: str
  messages: Iterable[anthropic_types.MessageParam]
  tools: list[anthropic_types.ToolParam]


def to_claude_role(role: Optional[str]) -> Literal["user", "assistant"]:
  if role in ["model", "assistant"]:
    return "assistant"
  return "user"


def to_google_genai_finish_reason(
    anthropic_stop_reason: Optional[str],
) -> types.FinishReason:
  if anthropic_stop_reason in ["end_turn", "stop_sequence", "tool_use"]:
    return "STOP"
  if anthropic_stop_reason == "max_tokens":
    return "MAX_TOKENS"
  return "FINISH_REASON_UNSPECIFIED"


def part_to_message_block(
    part: types.Part,
) -> Union[
    anthropic_types.TextBlockParam,
    anthropic_types.ImageBlockParam,
    anthropic_types.ToolUseBlockParam,
    anthropic_types.ToolResultBlockParam,
]:
  if part.text:
    return anthropic_types.TextBlockParam(text=part.text, type="text")
  if part.function_call:
    assert part.function_call.name

    return anthropic_types.ToolUseBlockParam(
        id=part.function_call.id or "",
        name=part.function_call.name,
        input=part.function_call.args,
        type="tool_use",
    )
  if part.function_response:
    content = ""
    if (
        "result" in part.function_response.response
        and part.function_response.response["result"]
    ):
      # Transformation is required because the content is a list of dict.
      # ToolResultBlockParam content doesn't support list of dict. Converting
      # to str to prevent anthropic.BadRequestError from being thrown.
      content = str(part.function_response.response["result"])
    return anthropic_types.ToolResultBlockParam(
        tool_use_id=part.function_response.id or "",
        type="tool_result",
        content=content,
        is_error=False,
    )
  raise NotImplementedError("Not supported yet.")


def content_to_message_param(
    content: types.Content,
) -> anthropic_types.MessageParam:
  return {
      "role": to_claude_role(content.role),
      "content": [part_to_message_block(part) for part in content.parts or []],
  }


def content_block_to_part(
    content_block: anthropic_types.ContentBlock,
) -> types.Part:
  if isinstance(content_block, anthropic_types.TextBlock):
    return types.Part.from_text(text=content_block.text)
  if isinstance(content_block, anthropic_types.ToolUseBlock):
    assert isinstance(content_block.input, dict)
    part = types.Part.from_function_call(
        name=content_block.name, args=content_block.input
    )
    part.function_call.id = content_block.id
    return part
  raise NotImplementedError("Not supported yet.")


def message_to_generate_content_response(
    message: anthropic_types.Message,
) -> LlmResponse:

  return LlmResponse(
      content=types.Content(
          role="model",
          parts=[content_block_to_part(cb) for cb in message.content],
      ),
      # TODO: Deal with these later.
      # finish_reason=to_google_genai_finish_reason(message.stop_reason),
      # usage_metadata=types.GenerateContentResponseUsageMetadata(
      #     prompt_token_count=message.usage.input_tokens,
      #     candidates_token_count=message.usage.output_tokens,
      #     total_token_count=(
      #         message.usage.input_tokens + message.usage.output_tokens
      #     ),
      # ),
  )


def _update_type_string(value_dict: dict[str, Any]):
  """Updates 'type' field to expected JSON schema format."""
  if "type" in value_dict:
    value_dict["type"] = value_dict["type"].lower()

  if "items" in value_dict:
    # 'type' field could exist for items as well, this would be the case if
    # items represent primitive types.
    _update_type_string(value_dict["items"])

    if "properties" in value_dict["items"]:
      # There could be properties as well on the items, especially if the items
      # are complex object themselves. We recursively traverse each individual
      # property as well and fix the "type" value.
      for _, value in value_dict["items"]["properties"].items():
        _update_type_string(value)


def function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> anthropic_types.ToolParam:
  assert function_declaration.name

  properties = {}
  if (
      function_declaration.parameters
      and function_declaration.parameters.properties
  ):
    for key, value in function_declaration.parameters.properties.items():
      value_dict = value.model_dump(exclude_none=True)
      _update_type_string(value_dict)
      properties[key] = value_dict

  return anthropic_types.ToolParam(
      name=function_declaration.name,
      description=function_declaration.description or "",
      input_schema={
          "type": "object",
          "properties": properties,
      },
  )


class Claude(BaseLlm):
  model: str = "claude-3-5-sonnet-v2@20241022"

  @staticmethod
  @override
  def supported_models() -> list[str]:
    return [r"claude-3-.*"]

  @override
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    messages = [
        content_to_message_param(content)
        for content in llm_request.contents or []
    ]
    tools = NOT_GIVEN
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
      tools = [
          function_declaration_to_tool_param(tool)
          for tool in llm_request.config.tools[0].function_declarations
      ]
    tool_choice = (
        anthropic_types.ToolChoiceAutoParam(
            type="auto",
            # TODO: allow parallel tool use.
            disable_parallel_tool_use=True,
        )
        if llm_request.tools_dict
        else NOT_GIVEN
    )
    message = self._anthropic_client.messages.create(
        model=llm_request.model,
        system=llm_request.config.system_instruction,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=MAX_TOKEN,
    )
    logger.info(
        "Claude response: %s",
        message.model_dump_json(indent=2, exclude_none=True),
    )
    yield message_to_generate_content_response(message)

  @cached_property
  def _anthropic_client(self) -> AnthropicVertex:
    if (
        "GOOGLE_CLOUD_PROJECT" not in os.environ
        or "GOOGLE_CLOUD_LOCATION" not in os.environ
    ):
      raise ValueError(
          "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for using"
          " Anthropic on Vertex."
      )

    return AnthropicVertex(
        project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
        region=os.environ["GOOGLE_CLOUD_LOCATION"],
    )



================================================
FILE: src/google/adk/models/base_llm.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from abc import abstractmethod
from typing import AsyncGenerator, TYPE_CHECKING

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict

from .base_llm_connection import BaseLlmConnection

if TYPE_CHECKING:
  from .llm_request import LlmRequest
  from .llm_response import LlmResponse


class BaseLlm(BaseModel):
  """The BaseLLM class.

  Attributes:
    model: The name of the LLM, e.g. gemini-1.5-flash or gemini-1.5-flash-001.
  """

  model_config = ConfigDict(
      # This allows us to use arbitrary types in the model. E.g. PIL.Image.
      arbitrary_types_allowed=True,
  )
  """The pydantic model config."""

  model: str
  """The name of the LLM, e.g. gemini-1.5-flash or gemini-1.5-flash-001."""

  @classmethod
  def supported_models(cls) -> list[str]:
    """Returns a list of supported models in regex for LlmRegistry."""
    return []

  @abstractmethod
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates one content from the given contents and tools.

    Args:
      llm_request: LlmRequest, the request to send to the LLM.
      stream: bool = False, whether to do streaming call.

    Yields:
      a generator of types.Content.

      For non-streaming call, it will only yield one Content.

      For streaming call, it may yield more than one content, but all yielded
      contents should be treated as one content by merging the
      parts list.
    """
    raise NotImplementedError(
        f'Async generation is not supported for {self.model}.'
    )
    yield  # AsyncGenerator requires a yield statement in function body.

  def _maybe_append_user_content(self, llm_request: LlmRequest):
    """Appends a user content, so that model can continue to output.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
    """
    # If no content is provided, append a user content to hint model response
    # using system instruction.
    if not llm_request.contents:
      llm_request.contents.append(
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      text=(
                          'Handle the requests as specified in the System'
                          ' Instruction.'
                      )
                  )
              ],
          )
      )
      return

    # Insert a user content to preserve user intent and to avoid empty
    # model response.
    if llm_request.contents[-1].role != 'user':
      llm_request.contents.append(
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      text=(
                          'Continue processing previous requests as instructed.'
                          ' Exit or provide a summary if no more outputs are'
                          ' needed.'
                      )
                  )
              ],
          )
      )

  def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
    """Creates a live connection to the LLM.

    Args:
      llm_request: LlmRequest, the request to send to the LLM.

    Returns:
      BaseLlmConnection, the connection to the LLM.
    """
    raise NotImplementedError(
        f'Live connection is not supported for {self.model}.'
    )



================================================
FILE: src/google/adk/models/base_llm_connection.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import AsyncGenerator
from google.genai import types
from .llm_response import LlmResponse


class BaseLlmConnection:
  """The base class for a live model connection."""

  @abstractmethod
  async def send_history(self, history: list[types.Content]):
    """Sends the conversation history to the model.

    You call this method right after setting up the model connection.
    The model will respond if the last content is from user, otherwise it will
    wait for new user input before responding.

    Args:
      history: The conversation history to send to the model.
    """
    pass

  @abstractmethod
  async def send_content(self, content: types.Content):
    """Sends a user content to the model.

    The model will respond immediately upon receiving the content.
    If you send function responses, all parts in the content should be function
    responses.

    Args:
      content: The content to send to the model.
    """
    pass

  @abstractmethod
  async def send_realtime(self, blob: types.Blob):
    """Sends a chunk of audio or a frame of video to the model in realtime.

    The model may not respond immediately upon receiving the blob. It will do
    voice activity detection and decide when to respond.

    Args:
      blob: The blob to send to the model.
    """
    pass

  @abstractmethod
  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Receives the model response using the llm server connection.

    Args: None.

    Yields:
      LlmResponse: The model response.
    """
    pass

  @abstractmethod
  async def close(self):
    """Closes the llm server connection."""
    pass



================================================
FILE: src/google/adk/models/gemini_llm_connection.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import AsyncGenerator

from google.genai import live
from google.genai import types

from .base_llm_connection import BaseLlmConnection
from .llm_response import LlmResponse

logger = logging.getLogger(__name__)


class GeminiLlmConnection(BaseLlmConnection):
  """The Gemini model connection."""

  def __init__(self, gemini_session: live.AsyncSession):
    self._gemini_session = gemini_session

  async def send_history(self, history: list[types.Content]):
    """Sends the conversation history to the gemini model.

    You call this method right after setting up the model connection.
    The model will respond if the last content is from user, otherwise it will
    wait for new user input before responding.

    Args:
      history: The conversation history to send to the model.
    """

    # TODO: Remove this filter and translate unary contents to streaming
    # contents properly.

    # We ignore any audio from user during the agent transfer phase
    contents = [
        content
        for content in history
        if content.parts and content.parts[0].text
    ]

    if contents:
      await self._gemini_session.send(
          input=types.LiveClientContent(
              turns=contents,
              turn_complete=contents[-1].role == 'user',
          ),
      )
    else:
      logger.info('no content is sent')

  async def send_content(self, content: types.Content):
    """Sends a user content to the gemini model.

    The model will respond immediately upon receiving the content.
    If you send function responses, all parts in the content should be function
    responses.

    Args:
      content: The content to send to the model.
    """

    assert content.parts
    if content.parts[0].function_response:
      # All parts have to be function responses.
      function_responses = [part.function_response for part in content.parts]
      logger.debug('Sending LLM function response: %s', function_responses)
      await self._gemini_session.send(
          input=types.LiveClientToolResponse(
              function_responses=function_responses
          ),
      )
    else:
      logger.debug('Sending LLM new content %s', content)
      await self._gemini_session.send(
          input=types.LiveClientContent(
              turns=[content],
              turn_complete=True,
          )
      )

  async def send_realtime(self, blob: types.Blob):
    """Sends a chunk of audio or a frame of video to the model in realtime.

    Args:
      blob: The blob to send to the model.
    """

    input_blob = blob.model_dump()
    logger.debug('Sending LLM Blob: %s', input_blob)
    await self._gemini_session.send(input=input_blob)

  def __build_full_text_response(self, text: str):
    """Builds a full text response.

    The text should not partial and the returned LlmResponse is not be
    partial.

    Args:
      text: The text to be included in the response.

    Returns:
      An LlmResponse containing the full text.
    """
    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[types.Part.from_text(text=text)],
        ),
    )

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Receives the model response using the llm server connection.

    Yields:
      LlmResponse: The model response.
    """

    text = ''
    async for message in self._gemini_session.receive():
      logger.debug('Got LLM Live message: %s', message)
      if message.server_content:
        content = message.server_content.model_turn
        if content and content.parts:
          llm_response = LlmResponse(
              content=content, interrupted=message.server_content.interrupted
          )
          if content.parts[0].text:
            text += content.parts[0].text
            llm_response.partial = True
          # don't yield the merged text event when receiving audio data
          elif text and not content.parts[0].inline_data:
            yield self.__build_full_text_response(text)
            text = ''
          yield llm_response
        if (
            message.server_content.input_transcription
            and message.server_content.input_transcription.text
        ):
            user_text = message.server_content.input_transcription.text
            parts = [
                types.Part.from_text(
                    text=user_text,
                )
            ]
            llm_response = LlmResponse(
                content=types.Content(role='user', parts=parts)
            )
            yield llm_response
        if (
            message.server_content.output_transcription
            and message.server_content.output_transcription.text
        ):
          # TODO: Right now, we just support output_transcription without
          # changing interface and data protocol. Later, we can consider to
          # support output_transcription as a separate field in LlmResponse.

          # Transcription is always considered as partial event
          # We rely on other control signals to determine when to yield the
          # full text response(turn_complete, interrupted, or tool_call).
          text += message.server_content.output_transcription.text
          parts = [
              types.Part.from_text(
                  text=message.server_content.output_transcription.text
              )
          ]
          llm_response = LlmResponse(
              content=types.Content(role='model', parts=parts), partial=True
          )
          yield llm_response

        if message.server_content.turn_complete:
          if text:
            yield self.__build_full_text_response(text)
            text = ''
          yield LlmResponse(
              turn_complete=True, interrupted=message.server_content.interrupted
          )
          break
        # in case of empty content or parts, we sill surface it
        # in case it's an interrupted message, we merge the previous partial
        # text. Other we don't merge. because content can be none when model
        # safety threshold is triggered
        if message.server_content.interrupted and text:
          yield self.__build_full_text_response(text)
          text = ''
        yield LlmResponse(interrupted=message.server_content.interrupted)
      if message.tool_call:
        if text:
          yield self.__build_full_text_response(text)
          text = ''
        parts = [
            types.Part(function_call=function_call)
            for function_call in message.tool_call.function_calls
        ]
        yield LlmResponse(content=types.Content(role='model', parts=parts))

  async def close(self):
    """Closes the llm server connection."""

    await self._gemini_session.close()



================================================
FILE: src/google/adk/models/google_llm.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import contextlib
from functools import cached_property
import logging
import sys
from typing import AsyncGenerator
from typing import cast
from typing import TYPE_CHECKING

from google.genai import Client
from google.genai import types
from typing_extensions import override

from .. import version
from .base_llm import BaseLlm
from .base_llm_connection import BaseLlmConnection
from .gemini_llm_connection import GeminiLlmConnection
from .llm_response import LlmResponse

if TYPE_CHECKING:
  from .llm_request import LlmRequest

logger = logging.getLogger(__name__)

_NEW_LINE = '\n'
_EXCLUDED_PART_FIELD = {'inline_data': {'data'}}


class Gemini(BaseLlm):
  """Integration for Gemini models.

  Attributes:
    model: The name of the Gemini model.
  """

  model: str = 'gemini-1.5-flash'

  @staticmethod
  @override
  def supported_models() -> list[str]:
    """Provides the list of supported models.

    Returns:
      A list of supported models.
    """

    return [
        r'gemini-.*',
        # fine-tuned vertex endpoint pattern
        r'projects\/.+\/locations\/.+\/endpoints\/.+',
        # vertex gemini long name
        r'projects\/.+\/locations\/.+\/publishers\/google\/models\/gemini.+',
    ]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Sends a request to the Gemini model.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """

    self._maybe_append_user_content(llm_request)
    logger.info(
        'Sending out request, model: %s, backend: %s, stream: %s',
        llm_request.model,
        self._api_backend,
        stream,
    )
    logger.info(_build_request_log(llm_request))

    if stream:
      responses = await self.api_client.aio.models.generate_content_stream(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      response = None
      text = ''
      # for sse, similar as bidi (see receive method in gemini_llm_connecton.py),
      # we need to mark those text content as partial and after all partial
      # contents are sent, we send an accumulated event which contains all the
      # previous partial content. The only difference is bidi rely on
      # complete_turn flag to detect end while sse depends on finish_reason.
      async for response in responses:
        logger.info(_build_response_log(response))
        llm_response = LlmResponse.create(response)
        if (
            llm_response.content
            and llm_response.content.parts
            and llm_response.content.parts[0].text
        ):
          text += llm_response.content.parts[0].text
          llm_response.partial = True
        elif text and (
            not llm_response.content
            or not llm_response.content.parts
            # don't yield the merged text event when receiving audio data
            or not llm_response.content.parts[0].inline_data
        ):
          yield LlmResponse(
              content=types.ModelContent(
                  parts=[types.Part.from_text(text=text)],
              ),
          )
          text = ''
        yield llm_response
      if (
          text
          and response
          and response.candidates
          and response.candidates[0].finish_reason == types.FinishReason.STOP
      ):
        yield LlmResponse(
            content=types.ModelContent(
                parts=[types.Part.from_text(text=text)],
            ),
        )

    else:
      response = await self.api_client.aio.models.generate_content(
          model=llm_request.model,
          contents=llm_request.contents,
          config=llm_request.config,
      )
      logger.info(_build_response_log(response))
      yield LlmResponse.create(response)

  @cached_property
  def api_client(self) -> Client:
    """Provides the api client.

    Returns:
      The api client.
    """
    return Client(
        http_options=types.HttpOptions(headers=self._tracking_headers)
    )

  @cached_property
  def _api_backend(self) -> str:
    return 'vertex' if self.api_client.vertexai else 'ml_dev'

  @cached_property
  def _tracking_headers(self) -> dict[str, str]:
    framework_label = f'google-adk/{version.__version__}'
    language_label = 'gl-python/' + sys.version.split()[0]
    version_header_value = f'{framework_label} {language_label}'
    tracking_headers = {
        'x-goog-api-client': version_header_value,
        'user-agent': version_header_value,
    }
    return tracking_headers

  @cached_property
  def _live_api_client(self) -> Client:
    if self._api_backend == 'vertex':
      # use default api version for vertex
      return Client(
          http_options=types.HttpOptions(headers=self._tracking_headers)
      )
    else:
      # use v1alpha for ml_dev
      api_version = 'v1alpha'
      return Client(
          http_options=types.HttpOptions(
              headers=self._tracking_headers, api_version=api_version
          )
      )

  @contextlib.asynccontextmanager
  async def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
    """Connects to the Gemini model and returns an llm connection.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.

    Yields:
      BaseLlmConnection, the connection to the Gemini model.
    """

    llm_request.live_connect_config.system_instruction = types.Content(
        role='system',
        parts=[
            types.Part.from_text(text=llm_request.config.system_instruction)
        ],
    )
    llm_request.live_connect_config.tools = llm_request.config.tools
    async with self._live_api_client.aio.live.connect(
        model=llm_request.model, config=llm_request.live_connect_config
    ) as live_session:
      yield GeminiLlmConnection(live_session)


def _build_function_declaration_log(
    func_decl: types.FunctionDeclaration,
) -> str:
  param_str = '{}'
  if func_decl.parameters and func_decl.parameters.properties:
    param_str = str({
        k: v.model_dump(exclude_none=True)
        for k, v in func_decl.parameters.properties.items()
    })
  return_str = 'None'
  if func_decl.response:
    return_str = str(func_decl.response.model_dump(exclude_none=True))
  return f'{func_decl.name}: {param_str} -> {return_str}'


def _build_request_log(req: LlmRequest) -> str:
  function_decls: list[types.FunctionDeclaration] = cast(
      list[types.FunctionDeclaration],
      req.config.tools[0].function_declarations if req.config.tools else [],
  )
  function_logs = (
      [
          _build_function_declaration_log(func_decl)
          for func_decl in function_decls
      ]
      if function_decls
      else []
  )
  contents_logs = [
      content.model_dump_json(
          exclude_none=True,
          exclude={
              'parts': {
                  i: _EXCLUDED_PART_FIELD for i in range(len(content.parts))
              }
          },
      )
      for content in req.contents
  ]

  return f"""
LLM Request:
-----------------------------------------------------------
System Instruction:
{req.config.system_instruction}
-----------------------------------------------------------
Contents:
{_NEW_LINE.join(contents_logs)}
-----------------------------------------------------------
Functions:
{_NEW_LINE.join(function_logs)}
-----------------------------------------------------------
"""


def _build_response_log(resp: types.GenerateContentResponse) -> str:
  function_calls_text = []
  if function_calls := resp.function_calls:
    for func_call in function_calls:
      function_calls_text.append(
          f'name: {func_call.name}, args: {func_call.args}'
      )
  return f"""
LLM Response:
-----------------------------------------------------------
Text:
{resp.text}
-----------------------------------------------------------
Function calls:
{_NEW_LINE.join(function_calls_text)}
-----------------------------------------------------------
Raw response:
{resp.model_dump_json(exclude_none=True)}
-----------------------------------------------------------
"""



================================================
FILE: src/google/adk/models/lite_llm.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import base64
import json
import logging
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

from google.genai import types
from litellm import acompletion
from litellm import ChatCompletionAssistantMessage
from litellm import ChatCompletionDeveloperMessage
from litellm import ChatCompletionImageUrlObject
from litellm import ChatCompletionMessageToolCall
from litellm import ChatCompletionTextObject
from litellm import ChatCompletionToolMessage
from litellm import ChatCompletionUserMessage
from litellm import ChatCompletionVideoUrlObject
from litellm import completion
from litellm import CustomStreamWrapper
from litellm import Function
from litellm import Message
from litellm import ModelResponse
from litellm import OpenAIMessageContent
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

logger = logging.getLogger(__name__)

_NEW_LINE = "\n"
_EXCLUDED_PART_FIELD = {"inline_data": {"data"}}


class FunctionChunk(BaseModel):
  id: Optional[str]
  name: Optional[str]
  args: Optional[str]


class TextChunk(BaseModel):
  text: str


class LiteLLMClient:
  """Provides acompletion method (for better testability)."""

  async def acompletion(
      self, model, messages, tools, **kwargs
  ) -> Union[ModelResponse, CustomStreamWrapper]:
    """Asynchronously calls acompletion.

    Args:
      model: The model name.
      messages: The messages to send to the model.
      tools: The tools to use for the model.
      **kwargs: Additional arguments to pass to acompletion.

    Returns:
      The model response as a message.
    """

    return await acompletion(
        model=model,
        messages=messages,
        tools=tools,
        **kwargs,
    )

  def completion(
      self, model, messages, tools, stream=False, **kwargs
  ) -> Union[ModelResponse, CustomStreamWrapper]:
    """Synchronously calls completion. This is used for streaming only.

    Args:
      model: The model to use.
      messages: The messages to send.
      tools: The tools to use for the model.
      stream: Whether to stream the response.
      **kwargs: Additional arguments to pass to completion.

    Returns:
      The response from the model.
    """

    return completion(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        **kwargs,
    )


def _safe_json_serialize(obj) -> str:
  """Convert any Python object to a JSON-serializable type or string.

  Args:
    obj: The object to serialize.

  Returns:
    The JSON-serialized object string or string.
  """

  try:
    # Try direct JSON serialization first
    return json.dumps(obj)
  except (TypeError, OverflowError):
    return str(obj)


def _content_to_message_param(
    content: types.Content,
) -> Union[Message, list[Message]]:
  """Converts a types.Content to a litellm Message or list of Messages.

  Handles multipart function responses by returning a list of
  ChatCompletionToolMessage objects if multiple function_response parts exist.

  Args:
    content: The content to convert.

  Returns:
    A litellm Message, a list of litellm Messages.
  """

  tool_messages = []
  for part in content.parts:
    if part.function_response:
      tool_messages.append(
          ChatCompletionToolMessage(
              role="tool",
              tool_call_id=part.function_response.id,
              content=_safe_json_serialize(part.function_response.response),
          )
      )
  if tool_messages:
    return tool_messages if len(tool_messages) > 1 else tool_messages[0]

  # Handle user or assistant messages
  role = _to_litellm_role(content.role)
  message_content = _get_content(content.parts) or None

  if role == "user":
    return ChatCompletionUserMessage(role="user", content=message_content)
  else:  # assistant/model
    tool_calls = []
    content_present = False
    for part in content.parts:
      if part.function_call:
        tool_calls.append(
            ChatCompletionMessageToolCall(
                type="function",
                id=part.function_call.id,
                function=Function(
                    name=part.function_call.name,
                    arguments=part.function_call.args,
                ),
            )
        )
      elif part.text or part.inline_data:
        content_present = True

    final_content = message_content if content_present else None

    return ChatCompletionAssistantMessage(
        role=role,
        content=final_content,
        tool_calls=tool_calls or None,
    )


def _get_content(
    parts: Iterable[types.Part],
) -> Union[OpenAIMessageContent, str]:
  """Converts a list of parts to litellm content.

  Args:
    parts: The parts to convert.

  Returns:
    The litellm content.
  """

  content_objects = []
  for part in parts:
    if part.text:
      if len(parts) == 1:
        return part.text
      content_objects.append(
          ChatCompletionTextObject(
              type="text",
              text=part.text,
          )
      )
    elif (
        part.inline_data
        and part.inline_data.data
        and part.inline_data.mime_type
    ):
      base64_string = base64.b64encode(part.inline_data.data).decode("utf-8")
      data_uri = f"data:{part.inline_data.mime_type};base64,{base64_string}"

      if part.inline_data.mime_type.startswith("image"):
        content_objects.append(
            ChatCompletionImageUrlObject(
                type="image_url",
                image_url=data_uri,
            )
        )
      elif part.inline_data.mime_type.startswith("video"):
        content_objects.append(
            ChatCompletionVideoUrlObject(
                type="video_url",
                video_url=data_uri,
            )
        )
      else:
        raise ValueError("LiteLlm(BaseLlm) does not support this content part.")

  return content_objects


def _to_litellm_role(role: Optional[str]) -> Literal["user", "assistant"]:
  """Converts a types.Content role to a litellm role.

  Args:
    role: The types.Content role.

  Returns:
    The litellm role.
  """

  if role in ["model", "assistant"]:
    return "assistant"
  return "user"


TYPE_LABELS = {
    "STRING": "string",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
    "OBJECT": "object",
    "ARRAY": "array",
    "INTEGER": "integer",
}


def _schema_to_dict(schema: types.Schema) -> dict:
  """Recursively converts a types.Schema to a dictionary.

  Args:
    schema: The schema to convert.

  Returns:
    The dictionary representation of the schema.
  """

  schema_dict = schema.model_dump(exclude_none=True)
  if "type" in schema_dict:
    schema_dict["type"] = schema_dict["type"].lower()
  if "items" in schema_dict:
    if isinstance(schema_dict["items"], dict):
      schema_dict["items"] = _schema_to_dict(
          types.Schema.model_validate(schema_dict["items"])
      )
    elif isinstance(schema_dict["items"]["type"], types.Type):
      schema_dict["items"]["type"] = TYPE_LABELS[
          schema_dict["items"]["type"].value
      ]
  if "properties" in schema_dict:
    properties = {}
    for key, value in schema_dict["properties"].items():
      if isinstance(value, types.Schema):
        properties[key] = _schema_to_dict(value)
      else:
        properties[key] = value
        if "type" in properties[key]:
          properties[key]["type"] = properties[key]["type"].lower()
    schema_dict["properties"] = properties
  return schema_dict


def _function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> dict:
  """Converts a types.FunctionDeclaration to a openapi spec dictionary.

  Args:
    function_declaration: The function declaration to convert.

  Returns:
    The openapi spec dictionary representation of the function declaration.
  """

  assert function_declaration.name

  properties = {}
  if (
      function_declaration.parameters
      and function_declaration.parameters.properties
  ):
    for key, value in function_declaration.parameters.properties.items():
      properties[key] = _schema_to_dict(value)

  return {
      "type": "function",
      "function": {
          "name": function_declaration.name,
          "description": function_declaration.description or "",
          "parameters": {
              "type": "object",
              "properties": properties,
          },
      },
  }


def _model_response_to_chunk(
    response: ModelResponse,
) -> Generator[
    Tuple[Optional[Union[TextChunk, FunctionChunk]], Optional[str]], None, None
]:
  """Converts a litellm message to text or function chunk.

  Args:
    response: The response from the model.

  Yields:
    A tuple of text or function chunk and finish reason.
  """

  message = None
  if response.get("choices", None):
    message = response["choices"][0].get("message", None)
    finish_reason = response["choices"][0].get("finish_reason", None)
    # check streaming delta
    if message is None and response["choices"][0].get("delta", None):
      message = response["choices"][0]["delta"]

    if message.get("content", None):
      yield TextChunk(text=message.get("content")), finish_reason

    if message.get("tool_calls", None):
      for tool_call in message.get("tool_calls"):
        # aggregate tool_call
        if tool_call.type == "function":
          yield FunctionChunk(
              id=tool_call.id,
              name=tool_call.function.name,
              args=tool_call.function.arguments,
          ), finish_reason

    if finish_reason and not (
        message.get("content", None) or message.get("tool_calls", None)
    ):
      yield None, finish_reason

  if not message:
    yield None, None


def _model_response_to_generate_content_response(
    response: ModelResponse,
) -> LlmResponse:
  """Converts a litellm response to LlmResponse.

  Args:
    response: The model response.

  Returns:
    The LlmResponse.
  """

  message = None
  if response.get("choices", None):
    message = response["choices"][0].get("message", None)

  if not message:
    raise ValueError("No message in response")
  return _message_to_generate_content_response(message)


def _message_to_generate_content_response(
    message: Message, is_partial: bool = False
) -> LlmResponse:
  """Converts a litellm message to LlmResponse.

  Args:
    message: The message to convert.
    is_partial: Whether the message is partial.

  Returns:
    The LlmResponse.
  """

  parts = []
  if message.get("content", None):
    parts.append(types.Part.from_text(text=message.get("content")))

  if message.get("tool_calls", None):
    for tool_call in message.get("tool_calls"):
      if tool_call.type == "function":
        part = types.Part.from_function_call(
            name=tool_call.function.name,
            args=json.loads(tool_call.function.arguments or "{}"),
        )
        part.function_call.id = tool_call.id
        parts.append(part)

  return LlmResponse(
      content=types.Content(role="model", parts=parts), partial=is_partial
  )


def _get_completion_inputs(
    llm_request: LlmRequest,
) -> tuple[Iterable[Message], Iterable[dict]]:
  """Converts an LlmRequest to litellm inputs.

  Args:
    llm_request: The LlmRequest to convert.

  Returns:
    The litellm inputs (message list and tool dictionary).
  """
  messages = []
  for content in llm_request.contents or []:
    message_param_or_list = _content_to_message_param(content)
    if isinstance(message_param_or_list, list):
      messages.extend(message_param_or_list)
    elif message_param_or_list:  # Ensure it's not None before appending
      messages.append(message_param_or_list)

  if llm_request.config.system_instruction:
    messages.insert(
        0,
        ChatCompletionDeveloperMessage(
            role="developer",
            content=llm_request.config.system_instruction,
        ),
    )

  tools = None
  if (
      llm_request.config
      and llm_request.config.tools
      and llm_request.config.tools[0].function_declarations
  ):
    tools = [
        _function_declaration_to_tool_param(tool)
        for tool in llm_request.config.tools[0].function_declarations
    ]
  return messages, tools


def _build_function_declaration_log(
    func_decl: types.FunctionDeclaration,
) -> str:
  """Builds a function declaration log.

  Args:
    func_decl: The function declaration to convert.

  Returns:
    The function declaration log.
  """

  param_str = "{}"
  if func_decl.parameters and func_decl.parameters.properties:
    param_str = str({
        k: v.model_dump(exclude_none=True)
        for k, v in func_decl.parameters.properties.items()
    })
  return_str = "None"
  if func_decl.response:
    return_str = str(func_decl.response.model_dump(exclude_none=True))
  return f"{func_decl.name}: {param_str} -> {return_str}"


def _build_request_log(req: LlmRequest) -> str:
  """Builds a request log.

  Args:
    req: The request to convert.

  Returns:
    The request log.
  """

  function_decls: list[types.FunctionDeclaration] = cast(
      list[types.FunctionDeclaration],
      req.config.tools[0].function_declarations if req.config.tools else [],
  )
  function_logs = (
      [
          _build_function_declaration_log(func_decl)
          for func_decl in function_decls
      ]
      if function_decls
      else []
  )
  contents_logs = [
      content.model_dump_json(
          exclude_none=True,
          exclude={
              "parts": {
                  i: _EXCLUDED_PART_FIELD for i in range(len(content.parts))
              }
          },
      )
      for content in req.contents
  ]

  return f"""
LLM Request:
-----------------------------------------------------------
System Instruction:
{req.config.system_instruction}
-----------------------------------------------------------
Contents:
{_NEW_LINE.join(contents_logs)}
-----------------------------------------------------------
Functions:
{_NEW_LINE.join(function_logs)}
-----------------------------------------------------------
"""


class LiteLlm(BaseLlm):
  """Wrapper around litellm.

  This wrapper can be used with any of the models supported by litellm. The
  environment variable(s) needed for authenticating with the model endpoint must
  be set prior to instantiating this class.

  Example usage:
  ```
  os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
  os.environ["VERTEXAI_LOCATION"] = "your-gcp-location"

  agent = Agent(
      model=LiteLlm(model="vertex_ai/claude-3-7-sonnet@20250219"),
      ...
  )
  ```

  Attributes:
    model: The name of the LiteLlm model.
    llm_client: The LLM client to use for the model.
  """

  llm_client: LiteLLMClient = Field(default_factory=LiteLLMClient)
  """The LLM client to use for the model."""

  _additional_args: Dict[str, Any] = None

  def __init__(self, model: str, **kwargs):
    """Initializes the LiteLlm class.

    Args:
      model: The name of the LiteLlm model.
      **kwargs: Additional arguments to pass to the litellm completion api.
    """
    super().__init__(model=model, **kwargs)
    self._additional_args = kwargs
    # preventing generation call with llm_client
    # and overriding messages, tools and stream which are managed internally
    self._additional_args.pop("llm_client", None)
    self._additional_args.pop("messages", None)
    self._additional_args.pop("tools", None)
    # public api called from runner determines to stream or not
    self._additional_args.pop("stream", None)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates content asynchronously.

    Args:
      llm_request: LlmRequest, the request to send to the LiteLlm model.
      stream: bool = False, whether to do streaming call.

    Yields:
      LlmResponse: The model response.
    """

    self._maybe_append_user_content(llm_request)
    logger.debug(_build_request_log(llm_request))

    messages, tools = _get_completion_inputs(llm_request)

    completion_args = {
        "model": self.model,
        "messages": messages,
        "tools": tools,
    }
    completion_args.update(self._additional_args)

    if stream:
      text = ""
      function_name = ""
      function_args = ""
      function_id = None
      completion_args["stream"] = True
      for part in self.llm_client.completion(**completion_args):
        for chunk, finish_reason in _model_response_to_chunk(part):
          if isinstance(chunk, FunctionChunk):
            if chunk.name:
              function_name += chunk.name
            if chunk.args:
              function_args += chunk.args
            function_id = chunk.id or function_id
          elif isinstance(chunk, TextChunk):
            text += chunk.text
            yield _message_to_generate_content_response(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    content=chunk.text,
                ),
                is_partial=True,
            )
          if finish_reason == "tool_calls" and function_id:
            yield _message_to_generate_content_response(
                ChatCompletionAssistantMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            type="function",
                            id=function_id,
                            function=Function(
                                name=function_name,
                                arguments=function_args,
                            ),
                        )
                    ],
                )
            )
            function_name = ""
            function_args = ""
            function_id = None
          elif finish_reason == "stop" and text:
            yield _message_to_generate_content_response(
                ChatCompletionAssistantMessage(role="assistant", content=text)
            )
            text = ""

    else:
      response = await self.llm_client.acompletion(**completion_args)
      yield _model_response_to_generate_content_response(response)

  @staticmethod
  @override
  def supported_models() -> list[str]:
    """Provides the list of supported models.

    LiteLlm supports all models supported by litellm. We do not keep track of
    these models here. So we return an empty list.

    Returns:
      A list of supported models.
    """

    return []



================================================
FILE: src/google/adk/models/llm_request.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..tools.base_tool import BaseTool


class LlmRequest(BaseModel):
  """LLM request class that allows passing in tools, output schema and system

  instructions to the model.

  Attributes:
    model: The model name.
    contents: The contents to send to the model.
    config: Additional config for the generate content request.
    tools_dict: The tools dictionary.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)
  """The pydantic model config."""

  model: Optional[str] = None
  """The model name."""

  contents: list[types.Content] = Field(default_factory=list)
  """The contents to send to the model."""

  config: Optional[types.GenerateContentConfig] = None
  live_connect_config: types.LiveConnectConfig = types.LiveConnectConfig()
  """Additional config for the generate content request.

  tools in generate_content_config should not be set.
  """
  tools_dict: dict[str, BaseTool] = Field(default_factory=dict, exclude=True)
  """The tools dictionary."""

  def append_instructions(self, instructions: list[str]) -> None:
    """Appends instructions to the system instruction.

    Args:
      instructions: The instructions to append.
    """

    if self.config.system_instruction:
      self.config.system_instruction += '\n\n' + '\n\n'.join(instructions)
    else:
      self.config.system_instruction = '\n\n'.join(instructions)

  def append_tools(self, tools: list[BaseTool]) -> None:
    """Appends tools to the request.

    Args:
      tools: The tools to append.
    """

    if not tools:
      return
    declarations = []
    for tool in tools:
      if isinstance(tool, BaseTool):
        declaration = tool._get_declaration()
      else:
        declaration = tool.get_declaration()
      if declaration:
        declarations.append(declaration)
        self.tools_dict[tool.name] = tool
    if declarations:
      self.config.tools.append(types.Tool(function_declarations=declarations))

  def set_output_schema(self, base_model: type[BaseModel]) -> None:
    """Sets the output schema for the request.

    Args:
      base_model: The pydantic base model to set the output schema to.
    """

    self.config.response_schema = base_model
    self.config.response_mime_type = 'application/json'



================================================
FILE: src/google/adk/models/llm_response.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Optional

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict


class LlmResponse(BaseModel):
  """LLM response class that provides the first candidate response from the

  model if available. Otherwise, returns error code and message.

  Attributes:
    content: The content of the response.
    grounding_metadata: The grounding metadata of the response.
    partial: Indicates whether the text content is part of a unfinished text
      stream. Only used for streaming mode and when the content is plain text.
    turn_complete: Indicates whether the response from the model is complete.
      Only used for streaming mode.
    error_code: Error code if the response is an error. Code varies by model.
    error_message: Error message if the response is an error.
    interrupted: Flag indicating that LLM was interrupted when generating the
      content. Usually it's due to user interruption during a bidi streaming.
    custom_metadata: The custom metadata of the LlmResponse.
  """

  model_config = ConfigDict(extra='forbid')
  """The pydantic model config."""

  content: Optional[types.Content] = None
  """The content of the response."""

  grounding_metadata: Optional[types.GroundingMetadata] = None
  """The grounding metadata of the response."""

  partial: Optional[bool] = None
  """Indicates whether the text content is part of a unfinished text stream.

  Only used for streaming mode and when the content is plain text.
  """

  turn_complete: Optional[bool] = None
  """Indicates whether the response from the model is complete.

  Only used for streaming mode.
  """

  error_code: Optional[str] = None
  """Error code if the response is an error. Code varies by model."""

  error_message: Optional[str] = None
  """Error message if the response is an error."""

  interrupted: Optional[bool] = None
  """Flag indicating that LLM was interrupted when generating the content.
  Usually it's due to user interruption during a bidi streaming.
  """

  custom_metadata: Optional[dict[str, Any]] = None
  """The custom metadata of the LlmResponse.

  An optional key-value pair to label an LlmResponse.

  NOTE: the entire dict must be JSON serializable.
  """

  @staticmethod
  def create(
      generate_content_response: types.GenerateContentResponse,
  ) -> 'LlmResponse':
    """Creates an LlmResponse from a GenerateContentResponse.

    Args:
      generate_content_response: The GenerateContentResponse to create the
        LlmResponse from.

    Returns:
      The LlmResponse.
    """

    if generate_content_response.candidates:
      candidate = generate_content_response.candidates[0]
      if candidate.content and candidate.content.parts:
        return LlmResponse(
            content=candidate.content,
            grounding_metadata=candidate.grounding_metadata,
        )
      else:
        return LlmResponse(
            error_code=candidate.finish_reason,
            error_message=candidate.finish_message,
        )
    else:
      if generate_content_response.prompt_feedback:
        prompt_feedback = generate_content_response.prompt_feedback
        return LlmResponse(
            error_code=prompt_feedback.block_reason,
            error_message=prompt_feedback.block_reason_message,
        )
      else:
        return LlmResponse(
            error_code='UNKNOWN_ERROR',
            error_message='Unknown error.',
        )



================================================
FILE: src/google/adk/models/registry.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The registry class for model."""

from __future__ import annotations

from functools import lru_cache
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .base_llm import BaseLlm

logger = logging.getLogger(__name__)


_llm_registry_dict: dict[str, type[BaseLlm]] = {}
"""Registry for LLMs.

Key is the regex that matches the model name.
Value is the class that implements the model.
"""


class LLMRegistry:
  """Registry for LLMs."""

  @staticmethod
  def new_llm(model: str) -> BaseLlm:
    """Creates a new LLM instance.

    Args:
        model: The model name.

    Returns:
        The LLM instance.
    """

    return LLMRegistry.resolve(model)(model=model)

  @staticmethod
  def _register(model_name_regex: str, llm_cls: type[BaseLlm]):
    """Registers a new LLM class.

    Args:
        model_name_regex: The regex that matches the model name.
        llm_cls: The class that implements the model.
    """

    if model_name_regex in _llm_registry_dict:
      logger.info(
          'Updating LLM class for %s from %s to %s',
          model_name_regex,
          _llm_registry_dict[model_name_regex],
          llm_cls,
      )

    _llm_registry_dict[model_name_regex] = llm_cls

  @staticmethod
  def register(llm_cls: type[BaseLlm]):
    """Registers a new LLM class.

    Args:
        llm_cls: The class that implements the model.
    """

    for regex in llm_cls.supported_models():
      LLMRegistry._register(regex, llm_cls)

  @staticmethod
  @lru_cache(maxsize=32)
  def resolve(model: str) -> type[BaseLlm]:
    """Resolves the model to a BaseLlm subclass.

    Args:
        model: The model name.

    Returns:
        The BaseLlm subclass.
    Raises:
        ValueError: If the model is not found.
    """

    for regex, llm_class in _llm_registry_dict.items():
      if re.compile(regex).fullmatch(model):
        return llm_class

    raise ValueError(f'Model {model} not found.')



================================================
FILE: src/google/adk/planners/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base_planner import BasePlanner
from .built_in_planner import BuiltInPlanner
from .plan_re_act_planner import PlanReActPlanner

__all__ = [
    'BasePlanner',
    'BuiltInPlanner',
    'PlanReActPlanner',
]



================================================
FILE: src/google/adk/planners/base_planner.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from abc import ABC
from typing import List
from typing import Optional

from google.genai import types

from ..agents.callback_context import CallbackContext
from ..agents.readonly_context import ReadonlyContext
from ..models.llm_request import LlmRequest


class BasePlanner(ABC):
  """Abstract base class for all planners.

  The planner allows the agent to generate plans for the queries to guide its
  action.
  """

  @abc.abstractmethod
  def build_planning_instruction(
      self,
      readonly_context: ReadonlyContext,
      llm_request: LlmRequest,
  ) -> Optional[str]:
    """Builds the system instruction to be appended to the LLM request for planning.

    Args:
        readonly_context: The readonly context of the invocation.
        llm_request: The LLM request. Readonly.

    Returns:
        The planning system instruction, or None if no instruction is needed.
    """
    pass

  @abc.abstractmethod
  def process_planning_response(
      self,
      callback_context: CallbackContext,
      response_parts: List[types.Part],
  ) -> Optional[List[types.Part]]:
    """Processes the LLM response for planning.

    Args:
        callback_context: The callback context of the invocation.
        response_parts: The LLM response parts. Readonly.

    Returns:
        The processed response parts, or None if no processing is needed.
    """
    pass



================================================
FILE: src/google/adk/planners/built_in_planner.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from typing import Optional

from google.genai import types
from typing_extensions import override

from ..agents.callback_context import CallbackContext
from ..agents.readonly_context import ReadonlyContext
from ..models.llm_request import LlmRequest
from .base_planner import BasePlanner


class BuiltInPlanner(BasePlanner):
  """The built-in planner that uses model's built-in thinking features.

  Attributes:
      thinking_config: Config for model built-in thinking features. An error
        will be returned if this field is set for models that don't support
        thinking.
  """

  thinking_config: types.ThinkingConfig
  """
  Config for model built-in thinking features. An error will be returned if this
  field is set for models that don't support thinking.
  """

  def __init__(self, *, thinking_config: types.ThinkingConfig):
    """Initializes the built-in planner.

    Args:
      thinking_config: Config for model built-in thinking features. An error
        will be returned if this field is set for models that don't support
        thinking.
    """
    self.thinking_config = thinking_config

  def apply_thinking_config(self, llm_request: LlmRequest) -> None:
    """Applies the thinking config to the LLM request.

    Args:
      llm_request: The LLM request to apply the thinking config to.
    """
    if self.thinking_config:
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.thinking_config = self.thinking_config

  @override
  def build_planning_instruction(
      self,
      readonly_context: ReadonlyContext,
      llm_request: LlmRequest,
  ) -> Optional[str]:
    return

  @override
  def process_planning_response(
      self,
      callback_context: CallbackContext,
      response_parts: List[types.Part],
  ) -> Optional[List[types.Part]]:
    return



================================================
FILE: src/google/adk/planners/plan_re_act_planner.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
from typing import Optional

from google.genai import types
from typing_extensions import override

from ..agents.callback_context import CallbackContext
from ..agents.readonly_context import ReadonlyContext
from ..models.llm_request import LlmRequest
from .base_planner import BasePlanner

PLANNING_TAG = '/*PLANNING*/'
REPLANNING_TAG = '/*REPLANNING*/'
REASONING_TAG = '/*REASONING*/'
ACTION_TAG = '/*ACTION*/'
FINAL_ANSWER_TAG = '/*FINAL_ANSWER*/'


class PlanReActPlanner(BasePlanner):
  """Plan-Re-Act planner that constrains the LLM response to generate a plan before any action/observation.

  Note: this planner does not require the model to support built-in thinking
  features or setting the thinking config.
  """

  @override
  def build_planning_instruction(
      self,
      readonly_context: ReadonlyContext,
      llm_request: LlmRequest,
  ) -> str:
    return self._build_nl_planner_instruction()

  @override
  def process_planning_response(
      self,
      callback_context: CallbackContext,
      response_parts: List[types.Part],
  ) -> Optional[List[types.Part]]:
    if not response_parts:
      return None

    preserved_parts = []
    first_fc_part_index = -1
    for i in range(len(response_parts)):
      # Stop at the first (group of) function calls.
      if response_parts[i].function_call:
        # Ignore and filter out function calls with empty names.
        if not response_parts[i].function_call.name:
          continue
        preserved_parts.append(response_parts[i])
        first_fc_part_index = i
        break

      # Split the response into reasoning and final answer parts.
      self._handle_non_function_call_parts(response_parts[i], preserved_parts)

    if first_fc_part_index > 0:
      j = first_fc_part_index + 1
      while j < len(response_parts):
        if response_parts[j].function_call:
          preserved_parts.append(response_parts[j])
          j += 1
        else:
          break

    return preserved_parts

  def _split_by_last_pattern(self, text, separator):
    """Splits the text by the last occurrence of the separator.

    Args:
      text: The text to split.
      separator: The separator to split on.

    Returns:
      A tuple containing the text before the last separator and the text after
      the last separator.
    """
    index = text.rfind(separator)
    if index == -1:
      return text, ''
    return text[: index + len(separator)], text[index + len(separator) :]

  def _handle_non_function_call_parts(
      self, response_part: types.Part, preserved_parts: list[types.Part]
  ):
    """Handles non-function-call parts of the response.

    Args:
      response_part: The response part to handle.
      preserved_parts: The mutable list of parts to store the processed parts
        in.
    """
    if response_part.text and FINAL_ANSWER_TAG in response_part.text:
      reasoning_text, final_answer_text = self._split_by_last_pattern(
          response_part.text, FINAL_ANSWER_TAG
      )
      if reasoning_text:
        reasoning_part = types.Part(text=reasoning_text)
        self._mark_as_thought(reasoning_part)
        preserved_parts.append(reasoning_part)
      if final_answer_text:
        preserved_parts.append(
            types.Part(
                text=final_answer_text,
            )
        )
    else:
      response_text = response_part.text or ''
      # If the part is a text part with a planning/reasoning/action tag,
      # label it as reasoning.
      if response_text and (
          any(
              response_text.startswith(tag)
              for tag in [
                  PLANNING_TAG,
                  REASONING_TAG,
                  ACTION_TAG,
                  REPLANNING_TAG,
              ]
          )
      ):
        self._mark_as_thought(response_part)
      preserved_parts.append(response_part)

  def _mark_as_thought(self, response_part: types.Part):
    """Marks the response part as thought.

    Args:
      response_part: The mutable response part to mark as thought.
    """
    if response_part.text:
      response_part.thought = True
    return

  def _build_nl_planner_instruction(self) -> str:
    """Builds the NL planner instruction for the Plan-Re-Act planner.

    Returns:
      NL planner system instruction.
    """

    high_level_preamble = f"""
When answering the question, try to leverage the available tools to gather the information instead of your memorized knowledge.

Follow this process when answering the question: (1) first come up with a plan in natural language text format; (2) Then use tools to execute the plan and provide reasoning between tool code snippets to make a summary of current state and next step. Tool code snippets and reasoning should be interleaved with each other. (3) In the end, return one final answer.

Follow this format when answering the question: (1) The planning part should be under {PLANNING_TAG}. (2) The tool code snippets should be under {ACTION_TAG}, and the reasoning parts should be under {REASONING_TAG}. (3) The final answer part should be under {FINAL_ANSWER_TAG}.
"""

    planning_preamble = f"""
Below are the requirements for the planning:
The plan is made to answer the user query if following the plan. The plan is coherent and covers all aspects of information from user query, and only involves the tools that are accessible by the agent. The plan contains the decomposed steps as a numbered list where each step should use one or multiple available tools. By reading the plan, you can intuitively know which tools to trigger or what actions to take.
If the initial plan cannot be successfully executed, you should learn from previous execution results and revise your plan. The revised plan should be be under {REPLANNING_TAG}. Then use tools to follow the new plan.
"""

    reasoning_preamble = """
Below are the requirements for the reasoning:
The reasoning makes a summary of the current trajectory based on the user query and tool outputs. Based on the tool outputs and plan, the reasoning also comes up with instructions to the next steps, making the trajectory closer to the final answer.
"""

    final_answer_preamble = """
Below are the requirements for the final answer:
The final answer should be precise and follow query formatting requirements. Some queries may not be answerable with the available tools and information. In those cases, inform the user why you cannot process their query and ask for more information.
"""

    # Only contains the requirements for custom tool/libraries.
    tool_code_without_python_libraries_preamble = """
Below are the requirements for the tool code:

**Custom Tools:** The available tools are described in the context and can be directly used.
- Code must be valid self-contained Python snippets with no imports and no references to tools or Python libraries that are not in the context.
- You cannot use any parameters or fields that are not explicitly defined in the APIs in the context.
- The code snippets should be readable, efficient, and directly relevant to the user query and reasoning steps.
- When using the tools, you should use the library name together with the function name, e.g., vertex_search.search().
- If Python libraries are not provided in the context, NEVER write your own code other than the function calls using the provided tools.
"""

    user_input_preamble = """
VERY IMPORTANT instruction that you MUST follow in addition to the above instructions:

You should ask for clarification if you need more information to answer the question.
You should prefer using the information available in the context instead of repeated tool use.
"""

    return '\n\n'.join([
        high_level_preamble,
        planning_preamble,
        reasoning_preamble,
        final_answer_preamble,
        tool_code_without_python_libraries_preamble,
        user_input_preamble,
    ])



================================================
FILE: src/google/adk/sessions/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

from .base_session_service import BaseSessionService
from .in_memory_session_service import InMemorySessionService
from .session import Session
from .state import State
from .vertex_ai_session_service import VertexAiSessionService

logger = logging.getLogger(__name__)


__all__ = [
    'BaseSessionService',
    'InMemorySessionService',
    'Session',
    'State',
    'VertexAiSessionService',
]

try:
  from .database_session_service import DatabaseSessionService

  __all__.append('DatabaseSessionService')
except ImportError:
  logger.debug(
      'DatabaseSessionService require sqlalchemy>=2.0, please ensure it is'
      ' installed correctly.'
  )



================================================
FILE: src/google/adk/sessions/_session_util.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for session service."""

import base64
from typing import Any, Optional

from google.genai import types


def encode_content(content: types.Content):
  """Encodes a content object to a JSON dictionary."""
  encoded_content = content.model_dump(exclude_none=True)
  for p in encoded_content["parts"]:
    if "inline_data" in p:
      p["inline_data"]["data"] = base64.b64encode(
          p["inline_data"]["data"]
      ).decode("utf-8")
  return encoded_content


def decode_content(
    content: Optional[dict[str, Any]],
) -> Optional[types.Content]:
  """Decodes a content object from a JSON dictionary."""
  if not content:
    return None
  for p in content["parts"]:
    if "inline_data" in p:
      p["inline_data"]["data"] = base64.b64decode(p["inline_data"]["data"])
  return types.Content.model_validate(content)



================================================
FILE: src/google/adk/sessions/base_session_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from ..events.event import Event
from .session import Session
from .state import State


class GetSessionConfig(BaseModel):
  """The configuration of getting a session."""

  num_recent_events: Optional[int] = None
  after_timestamp: Optional[float] = None


class ListSessionsResponse(BaseModel):
  """The response of listing sessions.

  The events and states are not set within each Session object.
  """

  sessions: list[Session] = Field(default_factory=list)


class ListEventsResponse(BaseModel):
  """The response of listing events in a session."""

  events: list[Event] = Field(default_factory=list)
  next_page_token: Optional[str] = None


class BaseSessionService(abc.ABC):
  """Base class for session services.

  The service provides a set of methods for managing sessions and events.
  """

  @abc.abstractmethod
  def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    """Creates a new session.

    Args:
      app_name: the name of the app.
      user_id: the id of the user.
      state: the initial state of the session.
      session_id: the client-provided id of the session. If not provided, a
        generated ID will be used.

    Returns:
      session: The newly created session instance.
    """
    pass

  @abc.abstractmethod
  def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    """Gets a session."""
    pass

  @abc.abstractmethod
  def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    """Lists all the sessions."""
    pass

  @abc.abstractmethod
  def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Deletes a session."""
    pass

  @abc.abstractmethod
  def list_events(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> ListEventsResponse:
    """Lists events in a session."""
    pass

  def close_session(self, *, session: Session):
    """Closes a session."""
    # TODO: determine whether we want to finalize the session here.
    pass

  def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event to a session object."""
    if event.partial:
      return event
    self.__update_session_state(session, event)
    session.events.append(event)
    return event

  def __update_session_state(self, session: Session, event: Event):
    """Updates the session state based on the event."""
    if not event.actions or not event.actions.state_delta:
      return
    for key, value in event.actions.state_delta.items():
      if key.startswith(State.TEMP_PREFIX):
        continue
      session.state.update({key: value})



================================================
FILE: src/google/adk/sessions/database_session_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from datetime import datetime
import json
import logging
from typing import Any, Optional
import uuid

from sqlalchemy import Boolean
from sqlalchemy import delete
from sqlalchemy import Dialect
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import func
from sqlalchemy import Text
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session as DatabaseSessionFactory
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import MetaData
from sqlalchemy.types import DateTime
from sqlalchemy.types import PickleType
from sqlalchemy.types import String
from sqlalchemy.types import TypeDecorator
from typing_extensions import override
from tzlocal import get_localzone

from ..events.event import Event
from . import _session_util
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListEventsResponse
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State


logger = logging.getLogger(__name__)

DEFAULT_MAX_KEY_LENGTH = 128
DEFAULT_MAX_VARCHAR_LENGTH = 256


class DynamicJSON(TypeDecorator):
  """A JSON-like type that uses JSONB on PostgreSQL and TEXT with JSON

  serialization for other databases.
  """

  impl = Text  # Default implementation is TEXT

  def load_dialect_impl(self, dialect: Dialect):
    if dialect.name == "postgresql":
      return dialect.type_descriptor(postgresql.JSONB)
    if dialect.name == "mysql":
      # Use LONGTEXT for MySQL to address the data too long issue
      return dialect.type_descriptor(mysql.LONGTEXT)
    return dialect.type_descriptor(Text)  # Default to Text for other dialects

  def process_bind_param(self, value, dialect: Dialect):
    if value is not None:
      if dialect.name == "postgresql":
        return value  # JSONB handles dict directly
      return json.dumps(value)  # Serialize to JSON string for TEXT
    return value

  def process_result_value(self, value, dialect: Dialect):
    if value is not None:
      if dialect.name == "postgresql":
        return value  # JSONB returns dict directly
      else:
        return json.loads(value)  # Deserialize from JSON string for TEXT
    return value


class Base(DeclarativeBase):
  """Base class for database tables."""

  pass


class StorageSession(Base):
  """Represents a session stored in the database."""

  __tablename__ = "sessions"

  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH),
      primary_key=True,
      default=lambda: str(uuid.uuid4()),
  )

  state: Mapped[MutableDict[str, Any]] = mapped_column(
      MutableDict.as_mutable(DynamicJSON), default={}
  )

  create_time: Mapped[DateTime] = mapped_column(DateTime(), default=func.now())
  update_time: Mapped[DateTime] = mapped_column(
      DateTime(), default=func.now(), onupdate=func.now()
  )

  storage_events: Mapped[list["StorageEvent"]] = relationship(
      "StorageEvent",
      back_populates="storage_session",
  )

  def __repr__(self):
    return f"<StorageSession(id={self.id}, update_time={self.update_time})>"


class StorageEvent(Base):
  """Represents an event stored in the database."""

  __tablename__ = "events"

  id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  session_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )

  invocation_id: Mapped[str] = mapped_column(String(DEFAULT_MAX_VARCHAR_LENGTH))
  author: Mapped[str] = mapped_column(String(DEFAULT_MAX_VARCHAR_LENGTH))
  branch: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_VARCHAR_LENGTH), nullable=True
  )
  timestamp: Mapped[DateTime] = mapped_column(DateTime(), default=func.now())
  content: Mapped[dict[str, Any]] = mapped_column(DynamicJSON, nullable=True)
  actions: Mapped[MutableDict[str, Any]] = mapped_column(PickleType)

  long_running_tool_ids_json: Mapped[Optional[str]] = mapped_column(
      Text, nullable=True
  )
  grounding_metadata: Mapped[dict[str, Any]] = mapped_column(
      DynamicJSON, nullable=True
  )
  partial: Mapped[bool] = mapped_column(Boolean, nullable=True)
  turn_complete: Mapped[bool] = mapped_column(Boolean, nullable=True)
  error_code: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_VARCHAR_LENGTH), nullable=True
  )
  error_message: Mapped[str] = mapped_column(String(1024), nullable=True)
  interrupted: Mapped[bool] = mapped_column(Boolean, nullable=True)

  storage_session: Mapped[StorageSession] = relationship(
      "StorageSession",
      back_populates="storage_events",
  )

  __table_args__ = (
      ForeignKeyConstraint(
          ["app_name", "user_id", "session_id"],
          ["sessions.app_name", "sessions.user_id", "sessions.id"],
          ondelete="CASCADE",
      ),
  )

  @property
  def long_running_tool_ids(self) -> set[str]:
    return (
        set(json.loads(self.long_running_tool_ids_json))
        if self.long_running_tool_ids_json
        else set()
    )

  @long_running_tool_ids.setter
  def long_running_tool_ids(self, value: set[str]):
    if value is None:
      self.long_running_tool_ids_json = None
    else:
      self.long_running_tool_ids_json = json.dumps(list(value))


class StorageAppState(Base):
  """Represents an app state stored in the database."""

  __tablename__ = "app_states"

  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  state: Mapped[MutableDict[str, Any]] = mapped_column(
      MutableDict.as_mutable(DynamicJSON), default={}
  )
  update_time: Mapped[DateTime] = mapped_column(
      DateTime(), default=func.now(), onupdate=func.now()
  )


class StorageUserState(Base):
  """Represents a user state stored in the database."""

  __tablename__ = "user_states"

  app_name: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
      String(DEFAULT_MAX_KEY_LENGTH), primary_key=True
  )
  state: Mapped[MutableDict[str, Any]] = mapped_column(
      MutableDict.as_mutable(DynamicJSON), default={}
  )
  update_time: Mapped[DateTime] = mapped_column(
      DateTime(), default=func.now(), onupdate=func.now()
  )


class DatabaseSessionService(BaseSessionService):
  """A session service that uses a database for storage."""

  def __init__(self, db_url: str):
    """
    Args:
        db_url: The database URL to connect to.
    """
    # 1. Create DB engine for db connection
    # 2. Create all tables based on schema
    # 3. Initialize all properties

    try:
      db_engine = create_engine(db_url)
    except Exception as e:
      if isinstance(e, ArgumentError):
        raise ValueError(
            f"Invalid database URL format or argument '{db_url}'."
        ) from e
      if isinstance(e, ImportError):
        raise ValueError(
            f"Database related module not found for URL '{db_url}'."
        ) from e
      raise ValueError(
          f"Failed to create database engine for URL '{db_url}'"
      ) from e

    # Get the local timezone
    local_timezone = get_localzone()
    logger.info(f"Local timezone: {local_timezone}")

    self.db_engine: Engine = db_engine
    self.metadata: MetaData = MetaData()
    self.inspector = inspect(self.db_engine)

    # DB session factory method
    self.DatabaseSessionFactory: sessionmaker[DatabaseSessionFactory] = (
        sessionmaker(bind=self.db_engine)
    )

    # Uncomment to recreate DB every time
    # Base.metadata.drop_all(self.db_engine)
    Base.metadata.create_all(self.db_engine)

  @override
  def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    # 1. Populate states.
    # 2. Build storage session object
    # 3. Add the object to the table
    # 4. Build the session object with generated id
    # 5. Return the session

    with self.DatabaseSessionFactory() as sessionFactory:

      # Fetch app and user states from storage
      storage_app_state = sessionFactory.get(StorageAppState, (app_name))
      storage_user_state = sessionFactory.get(
          StorageUserState, (app_name, user_id)
      )

      app_state = storage_app_state.state if storage_app_state else {}
      user_state = storage_user_state.state if storage_user_state else {}

      # Create state tables if not exist
      if not storage_app_state:
        storage_app_state = StorageAppState(app_name=app_name, state={})
        sessionFactory.add(storage_app_state)
      if not storage_user_state:
        storage_user_state = StorageUserState(
            app_name=app_name, user_id=user_id, state={}
        )
        sessionFactory.add(storage_user_state)

      # Extract state deltas
      app_state_delta, user_state_delta, session_state = _extract_state_delta(
          state
      )

      # Apply state delta
      app_state.update(app_state_delta)
      user_state.update(user_state_delta)

      # Store app and user state
      if app_state_delta:
        storage_app_state.state = app_state
      if user_state_delta:
        storage_user_state.state = user_state

      # Store the session
      storage_session = StorageSession(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=session_state,
      )
      sessionFactory.add(storage_session)
      sessionFactory.commit()

      sessionFactory.refresh(storage_session)

      # Merge states for response
      merged_state = _merge_state(app_state, user_state, session_state)
      session = Session(
          app_name=str(storage_session.app_name),
          user_id=str(storage_session.user_id),
          id=str(storage_session.id),
          state=merged_state,
          last_update_time=storage_session.update_time.timestamp(),
      )
      return session

  @override
  def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    # 1. Get the storage session entry from session table
    # 2. Get all the events based on session id and filtering config
    # 3. Convert and return the session
    with self.DatabaseSessionFactory() as sessionFactory:
      storage_session = sessionFactory.get(
          StorageSession, (app_name, user_id, session_id)
      )
      if storage_session is None:
        return None

      storage_events = (
          sessionFactory.query(StorageEvent)
          .filter(StorageEvent.session_id == storage_session.id)
          .filter(
              StorageEvent.timestamp < config.after_timestamp
              if config
              else True
          )
          .limit(config.num_recent_events if config else None)
          .order_by(StorageEvent.timestamp.asc())
          .all()
      )

      # Fetch states from storage
      storage_app_state = sessionFactory.get(StorageAppState, (app_name))
      storage_user_state = sessionFactory.get(
          StorageUserState, (app_name, user_id)
      )

      app_state = storage_app_state.state if storage_app_state else {}
      user_state = storage_user_state.state if storage_user_state else {}
      session_state = storage_session.state

      # Merge states
      merged_state = _merge_state(app_state, user_state, session_state)

      # Convert storage session to session
      session = Session(
          app_name=app_name,
          user_id=user_id,
          id=session_id,
          state=merged_state,
          last_update_time=storage_session.update_time.timestamp(),
      )
      session.events = [
          Event(
              id=e.id,
              author=e.author,
              branch=e.branch,
              invocation_id=e.invocation_id,
              content=_session_util.decode_content(e.content),
              actions=e.actions,
              timestamp=e.timestamp.timestamp(),
              long_running_tool_ids=e.long_running_tool_ids,
              grounding_metadata=e.grounding_metadata,
              partial=e.partial,
              turn_complete=e.turn_complete,
              error_code=e.error_code,
              error_message=e.error_message,
              interrupted=e.interrupted,
          )
          for e in storage_events
      ]
    return session

  @override
  def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    with self.DatabaseSessionFactory() as sessionFactory:
      results = (
          sessionFactory.query(StorageSession)
          .filter(StorageSession.app_name == app_name)
          .filter(StorageSession.user_id == user_id)
          .all()
      )
      sessions = []
      for storage_session in results:
        session = Session(
            app_name=app_name,
            user_id=user_id,
            id=storage_session.id,
            state={},
            last_update_time=storage_session.update_time.timestamp(),
        )
        sessions.append(session)
      return ListSessionsResponse(sessions=sessions)

  @override
  def delete_session(
      self, app_name: str, user_id: str, session_id: str
  ) -> None:
    with self.DatabaseSessionFactory() as sessionFactory:
      stmt = delete(StorageSession).where(
          StorageSession.app_name == app_name,
          StorageSession.user_id == user_id,
          StorageSession.id == session_id,
      )
      sessionFactory.execute(stmt)
      sessionFactory.commit()

  @override
  def append_event(self, session: Session, event: Event) -> Event:
    logger.info(f"Append event: {event} to session {session.id}")

    if event.partial:
      return event

    # 1. Check if timestamp is stale
    # 2. Update session attributes based on event config
    # 3. Store event to table
    with self.DatabaseSessionFactory() as sessionFactory:
      storage_session = sessionFactory.get(
          StorageSession, (session.app_name, session.user_id, session.id)
      )

      if storage_session.update_time.timestamp() > session.last_update_time:
        raise ValueError(
          f"Session last_update_time "
          f"{datetime.fromtimestamp(session.last_update_time):%Y-%m-%d %H:%M:%S} "
          f"is later than the update_time in storage "
          f"{storage_session.update_time:%Y-%m-%d %H:%M:%S}"
      )

      # Fetch states from storage
      storage_app_state = sessionFactory.get(
          StorageAppState, (session.app_name)
      )
      storage_user_state = sessionFactory.get(
          StorageUserState, (session.app_name, session.user_id)
      )

      app_state = storage_app_state.state if storage_app_state else {}
      user_state = storage_user_state.state if storage_user_state else {}
      session_state = storage_session.state

      # Extract state delta
      app_state_delta = {}
      user_state_delta = {}
      session_state_delta = {}
      if event.actions:
        if event.actions.state_delta:
          app_state_delta, user_state_delta, session_state_delta = (
              _extract_state_delta(event.actions.state_delta)
          )

      # Merge state
      app_state.update(app_state_delta)
      user_state.update(user_state_delta)
      session_state.update(session_state_delta)

      # Update storage
      storage_app_state.state = app_state
      storage_user_state.state = user_state
      storage_session.state = session_state

      storage_event = StorageEvent(
          id=event.id,
          invocation_id=event.invocation_id,
          author=event.author,
          branch=event.branch,
          actions=event.actions,
          session_id=session.id,
          app_name=session.app_name,
          user_id=session.user_id,
          timestamp=datetime.fromtimestamp(event.timestamp),
          long_running_tool_ids=event.long_running_tool_ids,
          grounding_metadata=event.grounding_metadata,
          partial=event.partial,
          turn_complete=event.turn_complete,
          error_code=event.error_code,
          error_message=event.error_message,
          interrupted=event.interrupted,
      )
      if event.content:
        storage_event.content = _session_util.encode_content(event.content)

      sessionFactory.add(storage_event)

      sessionFactory.commit()
      sessionFactory.refresh(storage_session)

      # Update timestamp with commit time
      session.last_update_time = storage_session.update_time.timestamp()

    # Also update the in-memory session
    super().append_event(session=session, event=event)
    return event

  @override
  def list_events(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> ListEventsResponse:
    raise NotImplementedError()


def convert_event(event: StorageEvent) -> Event:
  """Converts a storage event to an event."""
  return Event(
      id=event.id,
      author=event.author,
      branch=event.branch,
      invocation_id=event.invocation_id,
      content=event.content,
      actions=event.actions,
      timestamp=event.timestamp.timestamp(),
  )


def _extract_state_delta(state: dict[str, Any]):
  app_state_delta = {}
  user_state_delta = {}
  session_state_delta = {}
  if state:
    for key in state.keys():
      if key.startswith(State.APP_PREFIX):
        app_state_delta[key.removeprefix(State.APP_PREFIX)] = state[key]
      elif key.startswith(State.USER_PREFIX):
        user_state_delta[key.removeprefix(State.USER_PREFIX)] = state[key]
      elif not key.startswith(State.TEMP_PREFIX):
        session_state_delta[key] = state[key]
  return app_state_delta, user_state_delta, session_state_delta


def _merge_state(app_state, user_state, session_state):
  # Merge states for response
  merged_state = copy.deepcopy(session_state)
  for key in app_state.keys():
    merged_state[State.APP_PREFIX + key] = app_state[key]
  for key in user_state.keys():
    merged_state[State.USER_PREFIX + key] = user_state[key]
  return merged_state



================================================
FILE: src/google/adk/sessions/in_memory_session_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import time
from typing import Any
from typing import Optional
import uuid

from typing_extensions import override

from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListEventsResponse
from .base_session_service import ListSessionsResponse
from .session import Session
from .state import State


class InMemorySessionService(BaseSessionService):
  """An in-memory implementation of the session service."""

  def __init__(self):
    # A map from app name to a map from user ID to a map from session ID to session.
    self.sessions: dict[str, dict[str, dict[str, Session]]] = {}
    # A map from app name to a map from user ID to a map from key to the value.
    self.user_state: dict[str, dict[str, dict[str, Any]]] = {}
    # A map from app name to a map from key to the value.
    self.app_state: dict[str, dict[str, Any]] = {}

  @override
  def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    session_id = (
        session_id.strip()
        if session_id and session_id.strip()
        else str(uuid.uuid4())
    )
    session = Session(
        app_name=app_name,
        user_id=user_id,
        id=session_id,
        state=state or {},
        last_update_time=time.time(),
    )

    if app_name not in self.sessions:
      self.sessions[app_name] = {}
    if user_id not in self.sessions[app_name]:
      self.sessions[app_name][user_id] = {}
    self.sessions[app_name][user_id][session_id] = session

    copied_session = copy.deepcopy(session)
    return self._merge_state(app_name, user_id, copied_session)

  @override
  def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Session:
    if app_name not in self.sessions:
      return None
    if user_id not in self.sessions[app_name]:
      return None
    if session_id not in self.sessions[app_name][user_id]:
      return None

    session = self.sessions[app_name][user_id].get(session_id)
    copied_session = copy.deepcopy(session)

    if config:
      if config.num_recent_events:
        copied_session.events = copied_session.events[
            -config.num_recent_events :
        ]
      if config.after_timestamp:
        i = len(copied_session.events) - 1
        while i >= 0:
          if copied_session.events[i].timestamp < config.after_timestamp:
            break
          i -= 1
        if i >= 0:
          copied_session.events = copied_session.events[i + 1:]

    return self._merge_state(app_name, user_id, copied_session)

  def _merge_state(self, app_name: str, user_id: str, copied_session: Session):
    # Merge app state
    if app_name in self.app_state:
      for key in self.app_state[app_name].keys():
        copied_session.state[State.APP_PREFIX + key] = self.app_state[app_name][
            key
        ]

    if (
        app_name not in self.user_state
        or user_id not in self.user_state[app_name]
    ):
      return copied_session

    # Merge session state with user state.
    for key in self.user_state[app_name][user_id].keys():
      copied_session.state[State.USER_PREFIX + key] = self.user_state[app_name][
          user_id
      ][key]
    return copied_session

  @override
  def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    empty_response = ListSessionsResponse()
    if app_name not in self.sessions:
      return empty_response
    if user_id not in self.sessions[app_name]:
      return empty_response

    sessions_without_events = []
    for session in self.sessions[app_name][user_id].values():
      copied_session = copy.deepcopy(session)
      copied_session.events = []
      copied_session.state = {}
      sessions_without_events.append(copied_session)
    return ListSessionsResponse(sessions=sessions_without_events)

  @override
  def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    if (
        self.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        is None
    ):
      return None

    self.sessions[app_name][user_id].pop(session_id)

  @override
  def append_event(self, session: Session, event: Event) -> Event:
    # Update the in-memory session.
    super().append_event(session=session, event=event)
    session.last_update_time = event.timestamp

    # Update the storage session
    app_name = session.app_name
    user_id = session.user_id
    session_id = session.id
    if app_name not in self.sessions:
      return event
    if user_id not in self.sessions[app_name]:
      return event
    if session_id not in self.sessions[app_name][user_id]:
      return event

    if event.actions and event.actions.state_delta:
      for key in event.actions.state_delta:
        if key.startswith(State.APP_PREFIX):
          self.app_state.setdefault(app_name, {})[
              key.removeprefix(State.APP_PREFIX)
          ] = event.actions.state_delta[key]

        if key.startswith(State.USER_PREFIX):
          self.user_state.setdefault(app_name, {}).setdefault(user_id, {})[
              key.removeprefix(State.USER_PREFIX)
          ] = event.actions.state_delta[key]

    storage_session = self.sessions[app_name][user_id].get(session_id)
    super().append_event(session=storage_session, event=event)

    storage_session.last_update_time = event.timestamp

    return event

  @override
  def list_events(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> ListEventsResponse:
    raise NotImplementedError()



================================================
FILE: src/google/adk/sessions/session.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..events.event import Event


class Session(BaseModel):
  """Represents a series of interactions between a user and agents.

  Attributes:
    id: The unique identifier of the session.
    app_name: The name of the app.
    user_id: The id of the user.
    state: The state of the session.
    events: The events of the session, e.g. user input, model response, function
      call/response, etc.
    last_update_time: The last update time of the session.
  """

  model_config = ConfigDict(
      extra='forbid',
      arbitrary_types_allowed=True,
  )
  """The pydantic model config."""

  id: str
  """The unique identifier of the session."""
  app_name: str
  """The name of the app."""
  user_id: str
  """The id of the user."""
  state: dict[str, Any] = Field(default_factory=dict)
  """The state of the session."""
  events: list[Event] = Field(default_factory=list)
  """The events of the session, e.g. user input, model response, function
  call/response, etc."""
  last_update_time: float = 0.0
  """The last update time of the session."""



================================================
FILE: src/google/adk/sessions/state.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any


class State:
  """A state dict that maintain the current value and the pending-commit delta."""

  APP_PREFIX = "app:"
  USER_PREFIX = "user:"
  TEMP_PREFIX = "temp:"

  def __init__(self, value: dict[str, Any], delta: dict[str, Any]):
    """
    Args:
      value: The current value of the state dict.
      delta: The delta change to the current value that hasn't been committed.
    """
    self._value = value
    self._delta = delta

  def __getitem__(self, key: str) -> Any:
    """Returns the value of the state dict for the given key."""
    if key in self._delta:
      return self._delta[key]
    return self._value[key]

  def __setitem__(self, key: str, value: Any):
    """Sets the value of the state dict for the given key."""
    # TODO: make new change only store in delta, so that self._value is only
    #   updated at the storage commit time.
    self._value[key] = value
    self._delta[key] = value

  def __contains__(self, key: str) -> bool:
    """Whether the state dict contains the given key."""
    return key in self._value or key in self._delta

  def has_delta(self) -> bool:
    """Whether the state has pending delta."""
    return bool(self._delta)

  def get(self, key: str, default: Any = None) -> Any:
    """Returns the value of the state dict for the given key."""
    if key not in self:
      return default
    return self[key]

  def update(self, delta: dict[str, Any]):
    """Updates the state dict with the given delta."""
    self._value.update(delta)
    self._delta.update(delta)

  def to_dict(self) -> dict[str, Any]:
    """Returns the state dict."""
    result = {}
    result.update(self._value)
    result.update(self._delta)
    return result



================================================
FILE: src/google/adk/sessions/vertex_ai_session_service.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
import time
from typing import Any, Optional

from dateutil import parser
from google import genai
from typing_extensions import override

from ..events.event import Event
from ..events.event_actions import EventActions
from . import _session_util
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListEventsResponse
from .base_session_service import ListSessionsResponse
from .session import Session


isoparse = parser.isoparse
logger = logging.getLogger(__name__)


class VertexAiSessionService(BaseSessionService):
  """Connects to the managed Vertex AI Session Service."""

  def __init__(
      self,
      project: str = None,
      location: str = None,
  ):
    self.project = project
    self.location = location

    client = genai.Client(vertexai=True, project=project, location=location)
    self.api_client = client._api_client

  @override
  def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)

    session_json_dict = {'user_id': user_id}
    if state:
      session_json_dict['session_state'] = state

    api_response = self.api_client.request(
        http_method='POST',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions',
        request_dict=session_json_dict,
    )
    logger.info(f'Create Session response {api_response}')

    session_id = api_response['name'].split('/')[-3]
    operation_id = api_response['name'].split('/')[-1]

    max_retry_attempt = 5
    while max_retry_attempt >= 0:
      lro_response = self.api_client.request(
          http_method='GET',
          path=f'operations/{operation_id}',
          request_dict={},
      )

      if lro_response.get('done', None):
        break

      time.sleep(1)
      max_retry_attempt -= 1

    # Get session resource
    get_session_api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}',
        request_dict={},
    )

    update_timestamp = isoparse(
        get_session_api_response['updateTime']
    ).timestamp()
    session = Session(
        app_name=str(app_name),
        user_id=str(user_id),
        id=str(session_id),
        state=get_session_api_response.get('sessionState', {}),
        last_update_time=update_timestamp,
    )
    return session

  @override
  def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Session:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)

    # Get session resource
    get_session_api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}',
        request_dict={},
    )

    session_id = get_session_api_response['name'].split('/')[-1]
    update_timestamp = isoparse(
        get_session_api_response['updateTime']
    ).timestamp()
    session = Session(
        app_name=str(app_name),
        user_id=str(user_id),
        id=str(session_id),
        state=get_session_api_response.get('sessionState', {}),
        last_update_time=update_timestamp,
    )

    list_events_api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}/events',
        request_dict={},
    )

    # Handles empty response case
    if list_events_api_response.get('httpHeaders', None):
      return session

    session.events = [
        _from_api_event(event)
        for event in list_events_api_response['sessionEvents']
    ]
    session.events = [
        event for event in session.events if event.timestamp <= update_timestamp
    ]
    session.events.sort(key=lambda event: event.timestamp)

    if config:
      if config.num_recent_events:
        session.events = session.events[-config.num_recent_events :]
      elif config.after_timestamp:
        i = len(session.events) - 1
        while i >= 0:
          if session.events[i].timestamp < config.after_timestamp:
            break
          i -= 1
        if i >= 0:
          session.events = session.events[i:]

    return session

  @override
  def list_sessions(
      self, *, app_name: str, user_id: str
  ) -> ListSessionsResponse:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)

    api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions?filter=user_id={user_id}',
        request_dict={},
    )

    # Handles empty response case
    if api_response.get('httpHeaders', None):
      return ListSessionsResponse()

    sessions = []
    for api_session in api_response['sessions']:
      session = Session(
          app_name=app_name,
          user_id=user_id,
          id=api_session['name'].split('/')[-1],
          state={},
          last_update_time=isoparse(api_session['updateTime']).timestamp(),
      )
      sessions.append(session)
    return ListSessionsResponse(sessions=sessions)

  def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)
    self.api_client.request(
        http_method='DELETE',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}',
        request_dict={},
    )

  @override
  def list_events(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
  ) -> ListEventsResponse:
    reasoning_engine_id = _parse_reasoning_engine_id(app_name)
    api_response = self.api_client.request(
        http_method='GET',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session_id}/events',
        request_dict={},
    )

    logger.info(f'List events response {api_response}')

    # Handles empty response case
    if api_response.get('httpHeaders', None):
      return ListEventsResponse()

    session_events = api_response['sessionEvents']

    return ListEventsResponse(
        events=[_from_api_event(event) for event in session_events]
    )

  @override
  def append_event(self, session: Session, event: Event) -> Event:
    # Update the in-memory session.
    super().append_event(session=session, event=event)

    reasoning_engine_id = _parse_reasoning_engine_id(session.app_name)
    self.api_client.request(
        http_method='POST',
        path=f'reasoningEngines/{reasoning_engine_id}/sessions/{session.id}:appendEvent',
        request_dict=_convert_event_to_json(event),
    )

    return event


def _convert_event_to_json(event: Event):
  metadata_json = {
      'partial': event.partial,
      'turn_complete': event.turn_complete,
      'interrupted': event.interrupted,
      'branch': event.branch,
      'long_running_tool_ids': (
          list(event.long_running_tool_ids)
          if event.long_running_tool_ids
          else None
      ),
  }
  if event.grounding_metadata:
    metadata_json['grounding_metadata'] = event.grounding_metadata.model_dump(
        exclude_none=True
    )

  event_json = {
      'author': event.author,
      'invocation_id': event.invocation_id,
      'timestamp': {
          'seconds': int(event.timestamp),
          'nanos': int(
              (event.timestamp - int(event.timestamp)) * 1_000_000_000
          ),
      },
      'error_code': event.error_code,
      'error_message': event.error_message,
      'event_metadata': metadata_json,
  }

  if event.actions:
    actions_json = {
        'skip_summarization': event.actions.skip_summarization,
        'state_delta': event.actions.state_delta,
        'artifact_delta': event.actions.artifact_delta,
        'transfer_agent': event.actions.transfer_to_agent,
        'escalate': event.actions.escalate,
        'requested_auth_configs': event.actions.requested_auth_configs,
    }
    event_json['actions'] = actions_json
  if event.content:
    event_json['content'] = _session_util.encode_content(event.content)
  if event.error_code:
    event_json['error_code'] = event.error_code
  if event.error_message:
    event_json['error_message'] = event.error_message
  return event_json


def _from_api_event(api_event: dict) -> Event:
  event_actions = EventActions()
  if api_event.get('actions', None):
    event_actions = EventActions(
        skip_summarization=api_event['actions'].get('skipSummarization', None),
        state_delta=api_event['actions'].get('stateDelta', {}),
        artifact_delta=api_event['actions'].get('artifactDelta', {}),
        transfer_to_agent=api_event['actions'].get('transferAgent', None),
        escalate=api_event['actions'].get('escalate', None),
        requested_auth_configs=api_event['actions'].get(
            'requestedAuthConfigs', {}
        ),
    )

  event = Event(
      id=api_event['name'].split('/')[-1],
      invocation_id=api_event['invocationId'],
      author=api_event['author'],
      actions=event_actions,
      content=_session_util.decode_content(api_event.get('content', None)),
      timestamp=isoparse(api_event['timestamp']).timestamp(),
      error_code=api_event.get('errorCode', None),
      error_message=api_event.get('errorMessage', None),
  )

  if api_event.get('eventMetadata', None):
    long_running_tool_ids_list = api_event['eventMetadata'].get(
        'longRunningToolIds', None
    )
    event.partial = api_event['eventMetadata'].get('partial', None)
    event.turn_complete = api_event['eventMetadata'].get('turnComplete', None)
    event.interrupted = api_event['eventMetadata'].get('interrupted', None)
    event.branch = api_event['eventMetadata'].get('branch', None)
    event.grounding_metadata = api_event['eventMetadata'].get(
        'groundingMetadata', None
    )
    event.long_running_tool_ids = (
        set(long_running_tool_ids_list) if long_running_tool_ids_list else None
    )

  return event


def _parse_reasoning_engine_id(app_name: str):
  if app_name.isdigit():
    return app_name

  pattern = r'^projects/([a-zA-Z0-9-_]+)/locations/([a-zA-Z0-9-_]+)/reasoningEngines/(\d+)$'
  match = re.fullmatch(pattern, app_name)

  if not bool(match):
    raise ValueError(
        f'App name {app_name} is not valid. It should either be the full'
        ' ReasoningEngine resource name, or the reasoning engine id.'
    )

  return match.groups()[-1]



================================================
FILE: src/google/adk/tools/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=g-bad-import-order
from .base_tool import BaseTool

from ..auth.auth_tool import AuthToolArguments
from .apihub_tool.apihub_toolset import APIHubToolset
from .built_in_code_execution_tool import built_in_code_execution
from .google_search_tool import google_search
from .vertex_ai_search_tool import VertexAiSearchTool
from .example_tool import ExampleTool
from .exit_loop_tool import exit_loop
from .function_tool import FunctionTool
from .get_user_choice_tool import get_user_choice_tool as get_user_choice
from .load_artifacts_tool import load_artifacts_tool as load_artifacts
from .load_memory_tool import load_memory_tool as load_memory
from .long_running_tool import LongRunningFunctionTool
from .preload_memory_tool import preload_memory_tool as preload_memory
from .tool_context import ToolContext
from .transfer_to_agent_tool import transfer_to_agent


__all__ = [
    'APIHubToolset',
    'AuthToolArguments',
    'BaseTool',
    'built_in_code_execution',
    'google_search',
    'VertexAiSearchTool',
    'ExampleTool',
    'exit_loop',
    'FunctionTool',
    'get_user_choice',
    'load_artifacts',
    'load_memory',
    'LongRunningFunctionTool',
    'preload_memory',
    'ToolContext',
    'transfer_to_agent',
]



================================================
FILE: src/google/adk/tools/_automatic_function_calling_util.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forked from google3/third_party/py/google/genai/_automatic_function_calling_util.py temporarily."""

import inspect
from types import FunctionType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Union

from google.genai import types
import pydantic
from pydantic import BaseModel
from pydantic import create_model
from pydantic import fields as pydantic_fields

from . import function_parameter_parse_util

_py_type_2_schema_type = {
    'str': types.Type.STRING,
    'int': types.Type.INTEGER,
    'float': types.Type.NUMBER,
    'bool': types.Type.BOOLEAN,
    'string': types.Type.STRING,
    'integer': types.Type.INTEGER,
    'number': types.Type.NUMBER,
    'boolean': types.Type.BOOLEAN,
    'list': types.Type.ARRAY,
    'array': types.Type.ARRAY,
    'tuple': types.Type.ARRAY,
    'object': types.Type.OBJECT,
    'Dict': types.Type.OBJECT,
    'List': types.Type.ARRAY,
    'Tuple': types.Type.ARRAY,
    'Any': types.Type.TYPE_UNSPECIFIED,
}


def _get_fields_dict(func: Callable) -> Dict:
  param_signature = dict(inspect.signature(func).parameters)
  fields_dict = {
      name: (
          # 1. We infer the argument type here: use Any rather than None so
          # it will not try to auto-infer the type based on the default value.
          (
              param.annotation
              if param.annotation != inspect.Parameter.empty
              else Any
          ),
          pydantic.Field(
              # 2. We do not support default values for now.
              default=(
                  param.default
                  if param.default != inspect.Parameter.empty
                  # ! Need to use Undefined instead of None
                  else pydantic_fields.PydanticUndefined
              ),
              # 3. Do not support parameter description for now.
              description=None,
          ),
      )
      for name, param in param_signature.items()
      # We do not support *args or **kwargs
      if param.kind
      in (
          inspect.Parameter.POSITIONAL_OR_KEYWORD,
          inspect.Parameter.KEYWORD_ONLY,
          inspect.Parameter.POSITIONAL_ONLY,
      )
  }
  return fields_dict


def _annotate_nullable_fields(schema: Dict):
  for _, property_schema in schema.get('properties', {}).items():
    # for Optional[T], the pydantic schema is:
    # {
    #   "type": "object",
    #   "properties": {
    #     "anyOf": [
    #       {
    #         "type": "null"
    #       },
    #       {
    #         "type": "T"
    #       }
    #     ]
    #   }
    # }
    for type_ in property_schema.get('anyOf', []):
      if type_.get('type') == 'null':
        property_schema['nullable'] = True
        property_schema['anyOf'].remove(type_)
        break


def _annotate_required_fields(schema: Dict):
  required = [
      field_name
      for field_name, field_schema in schema.get('properties', {}).items()
      if not field_schema.get('nullable') and 'default' not in field_schema
  ]
  schema['required'] = required


def _remove_any_of(schema: Dict):
  for _, property_schema in schema.get('properties', {}).items():
    union_types = property_schema.pop('anyOf', None)
    # Take the first non-null type.
    if union_types:
      for type_ in union_types:
        if type_.get('type') != 'null':
          property_schema.update(type_)


def _remove_default(schema: Dict):
  for _, property_schema in schema.get('properties', {}).items():
    property_schema.pop('default', None)


def _remove_nullable(schema: Dict):
  for _, property_schema in schema.get('properties', {}).items():
    property_schema.pop('nullable', None)


def _remove_title(schema: Dict):
  for _, property_schema in schema.get('properties', {}).items():
    property_schema.pop('title', None)


def _get_pydantic_schema(func: Callable) -> Dict:
  fields_dict = _get_fields_dict(func)
  if 'tool_context' in fields_dict.keys():
    fields_dict.pop('tool_context')
  return pydantic.create_model(func.__name__, **fields_dict).model_json_schema()


def _process_pydantic_schema(vertexai: bool, schema: Dict) -> Dict:
  _annotate_nullable_fields(schema)
  _annotate_required_fields(schema)
  if not vertexai:
    _remove_any_of(schema)
    _remove_default(schema)
    _remove_nullable(schema)
    _remove_title(schema)
  return schema


def _map_pydantic_type_to_property_schema(property_schema: Dict):
  if 'type' in property_schema:
    property_schema['type'] = _py_type_2_schema_type.get(
        property_schema['type'], 'TYPE_UNSPECIFIED'
    )
    if property_schema['type'] == 'ARRAY':
      _map_pydantic_type_to_property_schema(property_schema['items'])
  for type_ in property_schema.get('anyOf', []):
    if 'type' in type_:
      type_['type'] = _py_type_2_schema_type.get(
          type_['type'], 'TYPE_UNSPECIFIED'
      )
      # TODO: To investigate. Unclear why a Type is needed with 'anyOf' to
      # avoid google.genai.errors.ClientError: 400 INVALID_ARGUMENT.
      property_schema['type'] = type_['type']


def _map_pydantic_type_to_schema_type(schema: Dict):
  for _, property_schema in schema.get('properties', {}).items():
    _map_pydantic_type_to_property_schema(property_schema)


def _get_return_type(func: Callable) -> Any:
  return _py_type_2_schema_type.get(
      inspect.signature(func).return_annotation.__name__,
      inspect.signature(func).return_annotation.__name__,
  )


def build_function_declaration(
    func: Union[Callable, BaseModel],
    ignore_params: Optional[list[str]] = None,
    variant: Literal['GOOGLE_AI', 'VERTEX_AI', 'DEFAULT'] = 'GOOGLE_AI',
) -> types.FunctionDeclaration:
  signature = inspect.signature(func)
  should_update_signature = False
  new_func = None
  if not ignore_params:
    ignore_params = []
  for name, _ in signature.parameters.items():
    if name in ignore_params:
      should_update_signature = True
      break
  if should_update_signature:
    new_params = [
        param
        for name, param in signature.parameters.items()
        if name not in ignore_params
    ]
    if isinstance(func, type):
      fields = {
          name: (param.annotation, param.default)
          for name, param in signature.parameters.items()
          if name not in ignore_params
      }
      new_func = create_model(func.__name__, **fields)
    else:
      new_sig = signature.replace(parameters=new_params)
      new_func = FunctionType(
          func.__code__,
          func.__globals__,
          func.__name__,
          func.__defaults__,
          func.__closure__,
      )
      new_func.__signature__ = new_sig

  return (
      from_function_with_options(func, variant)
      if not should_update_signature
      else from_function_with_options(new_func, variant)
  )


def build_function_declaration_for_langchain(
    vertexai: bool, name, description, func, param_pydantic_schema
) -> types.FunctionDeclaration:
  param_pydantic_schema = _process_pydantic_schema(
      vertexai, {'properties': param_pydantic_schema}
  )['properties']
  param_copy = param_pydantic_schema.copy()
  required_fields = param_copy.pop('required', [])
  before_param_pydantic_schema = {
      'properties': param_copy,
      'required': required_fields,
  }
  return build_function_declaration_util(
      vertexai, name, description, func, before_param_pydantic_schema
  )


def build_function_declaration_for_params_for_crewai(
    vertexai: bool, name, description, func, param_pydantic_schema
) -> types.FunctionDeclaration:
  param_pydantic_schema = _process_pydantic_schema(
      vertexai, param_pydantic_schema
  )
  param_copy = param_pydantic_schema.copy()
  return build_function_declaration_util(
      vertexai, name, description, func, param_copy
  )


def build_function_declaration_util(
    vertexai: bool, name, description, func, before_param_pydantic_schema
) -> types.FunctionDeclaration:
  _map_pydantic_type_to_schema_type(before_param_pydantic_schema)
  properties = before_param_pydantic_schema.get('properties', {})
  function_declaration = types.FunctionDeclaration(
      parameters=types.Schema(
          type='OBJECT',
          properties=properties,
      )
      if properties
      else None,
      description=description,
      name=name,
  )
  if vertexai and isinstance(func, Callable):
    return_pydantic_schema = _get_return_type(func)
    function_declaration.response = types.Schema(
        type=return_pydantic_schema,
    )
  return function_declaration


def from_function_with_options(
    func: Callable,
    variant: Literal['GOOGLE_AI', 'VERTEX_AI', 'DEFAULT'] = 'GOOGLE_AI',
) -> 'types.FunctionDeclaration':

  supported_variants = ['GOOGLE_AI', 'VERTEX_AI', 'DEFAULT']
  if variant not in supported_variants:
    raise ValueError(
        f'Unsupported variant: {variant}. Supported variants are:'
        f' {", ".join(supported_variants)}'
    )

  parameters_properties = {}
  for name, param in inspect.signature(func).parameters.items():
    if param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_ONLY,
    ):
      schema = function_parameter_parse_util._parse_schema_from_parameter(
          variant, param, func.__name__
      )
      parameters_properties[name] = schema
  declaration = types.FunctionDeclaration(
      name=func.__name__,
      description=func.__doc__,
  )
  if parameters_properties:
    declaration.parameters = types.Schema(
        type='OBJECT',
        properties=parameters_properties,
    )
    if variant == 'VERTEX_AI':
      declaration.parameters.required = (
          function_parameter_parse_util._get_required_fields(
              declaration.parameters
          )
      )
  if not variant == 'VERTEX_AI':
    return declaration

  return_annotation = inspect.signature(func).return_annotation
  if return_annotation is inspect._empty:
    return declaration

  declaration.response = (
      function_parameter_parse_util._parse_schema_from_parameter(
          variant,
          inspect.Parameter(
              'return_value',
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              annotation=return_annotation,
          ),
          func.__name__,
      )
  )
  return declaration



================================================
FILE: src/google/adk/tools/agent_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from pydantic import model_validator
from typing_extensions import override

from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..runners import Runner
from ..sessions.in_memory_session_service import InMemorySessionService
from . import _automatic_function_calling_util
from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from ..agents.llm_agent import LlmAgent


class AgentTool(BaseTool):
  """A tool that wraps an agent.

  This tool allows an agent to be called as a tool within a larger application.
  The agent's input schema is used to define the tool's input parameters, and
  the agent's output is returned as the tool's result.

  Attributes:
    agent: The agent to wrap.
    skip_summarization: Whether to skip summarization of the agent output.
  """

  def __init__(self, agent: BaseAgent, skip_summarization: bool = False):
    self.agent = agent
    self.skip_summarization: bool = skip_summarization

    super().__init__(name=agent.name, description=agent.description)

  @model_validator(mode='before')
  @classmethod
  def populate_name(cls, data: Any) -> Any:
    data['name'] = data['agent'].name
    return data

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    from ..agents.llm_agent import LlmAgent

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      result = _automatic_function_calling_util.build_function_declaration(
          func=self.agent.input_schema, variant=self._api_variant
      )
    else:
      result = types.FunctionDeclaration(
          parameters=types.Schema(
              type=types.Type.OBJECT,
              properties={
                  'request': types.Schema(
                      type=types.Type.STRING,
                  ),
              },
              required=['request'],
          ),
          description=self.agent.description,
          name=self.name,
      )
    result.name = self.name
    return result

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    from ..agents.llm_agent import LlmAgent

    if self.skip_summarization:
      tool_context.actions.skip_summarization = True

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      input_value = self.agent.input_schema.model_validate(args)
    else:
      input_value = args['request']

    if isinstance(self.agent, LlmAgent) and self.agent.input_schema:
      if isinstance(input_value, dict):
        input_value = self.agent.input_schema.model_validate(input_value)
      if not isinstance(input_value, self.agent.input_schema):
        raise ValueError(
            f'Input value {input_value} is not of type'
            f' `{self.agent.input_schema}`.'
        )
      content = types.Content(
          role='user',
          parts=[
              types.Part.from_text(
                  text=input_value.model_dump_json(exclude_none=True)
              )
          ],
      )
    else:
      content = types.Content(
          role='user',
          parts=[types.Part.from_text(text=input_value)],
      )
    runner = Runner(
        app_name=self.agent.name,
        agent=self.agent,
        # TODO(kech): Remove the access to the invocation context.
        #   It seems we don't need re-use artifact_service if we forward below.
        artifact_service=tool_context._invocation_context.artifact_service,
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
    )
    session = runner.session_service.create_session(
        app_name=self.agent.name,
        user_id='tmp_user',
        state=tool_context.state.to_dict(),
    )

    last_event = None
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
      # Forward state delta to parent session.
      if event.actions.state_delta:
        tool_context.state.update(event.actions.state_delta)
      last_event = event

    if runner.artifact_service:
      # Forward all artifacts to parent session.
      artifact_names = await runner.artifact_service.list_artifact_keys(
          app_name=session.app_name,
          user_id=session.user_id,
          session_id=session.id,
      )
      for artifact_name in artifact_names:
        if artifact := await runner.artifact_service.load_artifact(
            app_name=session.app_name,
            user_id=session.user_id,
            session_id=session.id,
            filename=artifact_name,
        ):
          await tool_context.save_artifact(
              filename=artifact_name, artifact=artifact
          )

    if (
        not last_event
        or not last_event.content
        or not last_event.content.parts
        or not last_event.content.parts[0].text
    ):
      return ''
    if isinstance(self.agent, LlmAgent) and self.agent.output_schema:
      tool_result = self.agent.output_schema.model_validate_json(
          last_event.content.parts[0].text
      ).model_dump(exclude_none=True)
    else:
      tool_result = last_event.content.parts[0].text
    return tool_result



================================================
FILE: src/google/adk/tools/base_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABC
import os
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from deprecated import deprecated
from google.genai import types

from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest


class BaseTool(ABC):
  """The base class for all tools."""

  name: str
  """The name of the tool."""
  description: str
  """The description of the tool."""

  is_long_running: bool = False
  """Whether the tool is a long running operation, which typically returns a
  resource id first and finishes the operation later."""

  def __init__(self, *, name, description, is_long_running: bool = False):
    self.name = name
    self.description = description
    self.is_long_running = is_long_running

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Gets the OpenAPI specification of this tool in the form of a FunctionDeclaration.

    NOTE
    - Required if subclass uses the default implementation of
      `process_llm_request` to add function declaration to LLM request.
    - Otherwise, can be skipped, e.g. for a built-in GoogleSearch tool for
      Gemini.

    Returns:
      The FunctionDeclaration of this tool, or None if it doesn't need to be
      added to LlmRequest.config.
    """
    return None

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Runs the tool with the given arguments and context.

    NOTE
    - Required if this tool needs to run at the client side.
    - Otherwise, can be skipped, e.g. for a built-in GoogleSearch tool for
      Gemini.

    Args:
      args: The LLM-filled arguments.
      tool_context: The context of the tool.

    Returns:
      The result of running the tool.
    """
    raise NotImplementedError(f'{type(self)} is not implemented')

  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Processes the outgoing LLM request for this tool.

    Use cases:
    - Most common use case is adding this tool to the LLM request.
    - Some tools may just preprocess the LLM request before it's sent out.

    Args:
      tool_context: The context of the tool.
      llm_request: The outgoing LLM request, mutable this method.
    """
    if (function_declaration := self._get_declaration()) is None:
      return

    llm_request.tools_dict[self.name] = self
    if tool_with_function_declarations := _find_tool_with_function_declarations(
        llm_request
    ):
      if tool_with_function_declarations.function_declarations is None:
        tool_with_function_declarations.function_declarations = []
      tool_with_function_declarations.function_declarations.append(
          function_declaration
      )
    else:
      llm_request.config = (
          types.GenerateContentConfig()
          if not llm_request.config
          else llm_request.config
      )
      llm_request.config.tools = (
          [] if not llm_request.config.tools else llm_request.config.tools
      )
      llm_request.config.tools.append(
          types.Tool(function_declarations=[function_declaration])
      )

  @property
  def _api_variant(self) -> str:
    use_vertexai = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '0').lower() in [
        'true',
        '1',
    ]
    return 'VERTEX_AI' if use_vertexai else 'GOOGLE_AI'


def _find_tool_with_function_declarations(
    llm_request: LlmRequest,
) -> Optional[types.Tool]:
  # TODO: add individual tool with declaration and merge in google_llm.py
  if not llm_request.config or not llm_request.config.tools:
    return None

  return next(
      (
          tool
          for tool in llm_request.config.tools
          if isinstance(tool, types.Tool) and tool.function_declarations
      ),
      None,
  )



================================================
FILE: src/google/adk/tools/built_in_code_execution_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class BuiltInCodeExecutionTool(BaseTool):
  """A built-in code execution tool that is automatically invoked by Gemini 2 models.

  This tool operates internally within the model and does not require or perform
  local code execution.
  """

  def __init__(self):
    # Name and description are not used because this is a model built-in tool.
    super().__init__(name='code_execution', description='code_execution')

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    if llm_request.model and llm_request.model.startswith('gemini-2'):
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.tools = llm_request.config.tools or []
      llm_request.config.tools.append(
          types.Tool(code_execution=types.ToolCodeExecution())
      )
    else:
      raise ValueError(
          f'Code execution tool is not supported for model {llm_request.model}'
      )


built_in_code_execution = BuiltInCodeExecutionTool()



================================================
FILE: src/google/adk/tools/crewai_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from google.genai import types
from typing_extensions import override

from . import _automatic_function_calling_util
from .function_tool import FunctionTool

try:
  from crewai.tools import BaseTool as CrewaiBaseTool
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        "Crewai Tools require Python 3.10+. Please upgrade your Python version."
    ) from e
  else:
    raise ImportError(
        "Crewai Tools require pip install 'google-adk[extensions]'."
    ) from e


class CrewaiTool(FunctionTool):
  """Use this class to wrap a CrewAI tool.

  If the original tool name and description are not suitable, you can override
  them in the constructor.
  """

  tool: CrewaiBaseTool
  """The wrapped CrewAI tool."""

  def __init__(self, tool: CrewaiBaseTool, *, name: str, description: str):
    super().__init__(tool.run)
    self.tool = tool
    if name:
      self.name = name
    elif tool.name:
      # Right now, CrewAI tool name contains white spaces. White spaces are
      # not supported in our framework. So we replace them with "_".
      self.name = tool.name.replace(" ", "_").lower()
    if description:
      self.description = description
    elif tool.description:
      self.description = tool.description

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    """Build the function declaration for the tool."""
    function_declaration = _automatic_function_calling_util.build_function_declaration_for_params_for_crewai(
        False,
        self.name,
        self.description,
        self.func,
        self.tool.args_schema.model_json_schema(),
    )
    return function_declaration



================================================
FILE: src/google/adk/tools/example_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Union

from pydantic import TypeAdapter
from typing_extensions import override

from ..examples import example_util
from ..examples.base_example_provider import BaseExampleProvider
from ..examples.example import Example
from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest


class ExampleTool(BaseTool):
  """A tool that adds (few-shot) examples to the LLM request.

  Attributes:
    examples: The examples to add to the LLM request.
  """

  def __init__(self, examples: Union[list[Example], BaseExampleProvider]):
    # Name and description are not used because this tool only changes
    # llm_request.
    super().__init__(name='example_tool', description='example tool')
    self.examples = (
        TypeAdapter(list[Example]).validate_python(examples)
        if isinstance(examples, list)
        else examples
    )

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    parts = tool_context.user_content.parts
    if not parts or not parts[0].text:
      return

    llm_request.append_instructions([
        example_util.build_example_si(
            self.examples, parts[0].text, llm_request.model
        )
    ])



================================================
FILE: src/google/adk/tools/exit_loop_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .tool_context import ToolContext


def exit_loop(tool_context: ToolContext):
  """Exits the loop.

  Call this function only when you are instructed to do so.
  """
  tool_context.actions.escalate = True



================================================
FILE: src/google/adk/tools/function_parameter_parse_util.py
================================================
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect
import logging
import types as typing_types
from typing import _GenericAlias
from typing import Any
from typing import get_args
from typing import get_origin
from typing import Literal
from typing import Union

from google.genai import types
import pydantic

_py_builtin_type_to_schema_type = {
    str: types.Type.STRING,
    int: types.Type.INTEGER,
    float: types.Type.NUMBER,
    bool: types.Type.BOOLEAN,
    list: types.Type.ARRAY,
    dict: types.Type.OBJECT,
}

logger = logging.getLogger(__name__)


def _is_builtin_primitive_or_compound(
    annotation: inspect.Parameter.annotation,
) -> bool:
  return annotation in _py_builtin_type_to_schema_type.keys()


def _raise_for_any_of_if_mldev(schema: types.Schema):
  if schema.any_of:
    raise ValueError(
        'AnyOf is not supported in function declaration schema for Google AI.'
    )


def _update_for_default_if_mldev(schema: types.Schema):
  if schema.default is not None:
    # TODO(kech): Remove this workaround once mldev supports default value.
    schema.default = None
    logger.warning(
        'Default value is not supported in function declaration schema for'
        ' Google AI.'
    )


def _raise_if_schema_unsupported(variant: str, schema: types.Schema):
  if not variant == 'VERTEX_AI':
    _raise_for_any_of_if_mldev(schema)
    _update_for_default_if_mldev(schema)


def _is_default_value_compatible(
    default_value: Any, annotation: inspect.Parameter.annotation
) -> bool:
  # None type is expected to be handled external to this function
  if _is_builtin_primitive_or_compound(annotation):
    return isinstance(default_value, annotation)

  if (
      isinstance(annotation, _GenericAlias)
      or isinstance(annotation, typing_types.GenericAlias)
      or isinstance(annotation, typing_types.UnionType)
  ):
    origin = get_origin(annotation)
    if origin in (Union, typing_types.UnionType):
      return any(
          _is_default_value_compatible(default_value, arg)
          for arg in get_args(annotation)
      )

    if origin is dict:
      return isinstance(default_value, dict)

    if origin is list:
      if not isinstance(default_value, list):
        return False
      # most tricky case, element in list is union type
      # need to apply any logic within all
      # see test case test_generic_alias_complex_array_with_default_value
      # a: typing.List[int | str | float | bool]
      # default_value: [1, 'a', 1.1, True]
      return all(
          any(
              _is_default_value_compatible(item, arg)
              for arg in get_args(annotation)
          )
          for item in default_value
      )

    if origin is Literal:
      return default_value in get_args(annotation)

  # return False for any other unrecognized annotation
  # let caller handle the raise
  return False


def _parse_schema_from_parameter(
    variant: str, param: inspect.Parameter, func_name: str
) -> types.Schema:
  """parse schema from parameter.

  from the simplest case to the most complex case.
  """
  schema = types.Schema()
  default_value_error_msg = (
      f'Default value {param.default} of parameter {param} of function'
      f' {func_name} is not compatible with the parameter annotation'
      f' {param.annotation}.'
  )
  if _is_builtin_primitive_or_compound(param.annotation):
    if param.default is not inspect.Parameter.empty:
      if not _is_default_value_compatible(param.default, param.annotation):
        raise ValueError(default_value_error_msg)
      schema.default = param.default
    schema.type = _py_builtin_type_to_schema_type[param.annotation]
    _raise_if_schema_unsupported(variant, schema)
    return schema
  if (
      get_origin(param.annotation) is Union
      # only parse simple UnionType, example int | str | float | bool
      # complex types.UnionType will be invoked in raise branch
      and all(
          (_is_builtin_primitive_or_compound(arg) or arg is type(None))
          for arg in get_args(param.annotation)
      )
  ):
    schema.type = types.Type.OBJECT
    schema.any_of = []
    unique_types = set()
    for arg in get_args(param.annotation):
      if arg.__name__ == 'NoneType':  # Optional type
        schema.nullable = True
        continue
      schema_in_any_of = _parse_schema_from_parameter(
          variant,
          inspect.Parameter(
              'item', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=arg
          ),
          func_name,
      )
      if (
          schema_in_any_of.model_dump_json(exclude_none=True)
          not in unique_types
      ):
        schema.any_of.append(schema_in_any_of)
        unique_types.add(schema_in_any_of.model_dump_json(exclude_none=True))
    if len(schema.any_of) == 1:  # param: list | None -> Array
      schema.type = schema.any_of[0].type
      schema.any_of = None
    if (
        param.default is not inspect.Parameter.empty
        and param.default is not None
    ):
      if not _is_default_value_compatible(param.default, param.annotation):
        raise ValueError(default_value_error_msg)
      schema.default = param.default
    _raise_if_schema_unsupported(variant, schema)
    return schema
  if isinstance(param.annotation, _GenericAlias) or isinstance(
      param.annotation, typing_types.GenericAlias
  ):
    origin = get_origin(param.annotation)
    args = get_args(param.annotation)
    if origin is dict:
      schema.type = types.Type.OBJECT
      if param.default is not inspect.Parameter.empty:
        if not _is_default_value_compatible(param.default, param.annotation):
          raise ValueError(default_value_error_msg)
        schema.default = param.default
      _raise_if_schema_unsupported(variant, schema)
      return schema
    if origin is Literal:
      if not all(isinstance(arg, str) for arg in args):
        raise ValueError(
            f'Literal type {param.annotation} must be a list of strings.'
        )
      schema.type = types.Type.STRING
      schema.enum = list(args)
      if param.default is not inspect.Parameter.empty:
        if not _is_default_value_compatible(param.default, param.annotation):
          raise ValueError(default_value_error_msg)
        schema.default = param.default
      _raise_if_schema_unsupported(variant, schema)
      return schema
    if origin is list:
      schema.type = types.Type.ARRAY
      schema.items = _parse_schema_from_parameter(
          variant,
          inspect.Parameter(
              'item',
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              annotation=args[0],
          ),
          func_name,
      )
      if param.default is not inspect.Parameter.empty:
        if not _is_default_value_compatible(param.default, param.annotation):
          raise ValueError(default_value_error_msg)
        schema.default = param.default
      _raise_if_schema_unsupported(variant, schema)
      return schema
    if origin is Union:
      schema.any_of = []
      schema.type = types.Type.OBJECT
      unique_types = set()
      for arg in args:
        if arg.__name__ == 'NoneType':  # Optional type
          schema.nullable = True
          continue
        schema_in_any_of = _parse_schema_from_parameter(
            variant,
            inspect.Parameter(
                'item',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=arg,
            ),
            func_name,
        )
        if (
            len(param.annotation.__args__) == 2
            and type(None) in param.annotation.__args__
        ):  # Optional type
          for optional_arg in param.annotation.__args__:
            if (
                hasattr(optional_arg, '__origin__')
                and optional_arg.__origin__ is list
            ):
              # Optional type with list, for example Optional[list[str]]
              schema.items = schema_in_any_of.items
        if (
            schema_in_any_of.model_dump_json(exclude_none=True)
            not in unique_types
        ):
          schema.any_of.append(schema_in_any_of)
          unique_types.add(schema_in_any_of.model_dump_json(exclude_none=True))
      if len(schema.any_of) == 1:  # param: Union[List, None] -> Array
        schema.type = schema.any_of[0].type
        schema.any_of = None
      if (
          param.default is not None
          and param.default is not inspect.Parameter.empty
      ):
        if not _is_default_value_compatible(param.default, param.annotation):
          raise ValueError(default_value_error_msg)
        schema.default = param.default
      _raise_if_schema_unsupported(variant, schema)
      return schema
      # all other generic alias will be invoked in raise branch
  if (
      inspect.isclass(param.annotation)
      # for user defined class, we only support pydantic model
      and issubclass(param.annotation, pydantic.BaseModel)
  ):
    if (
        param.default is not inspect.Parameter.empty
        and param.default is not None
    ):
      schema.default = param.default
    schema.type = types.Type.OBJECT
    schema.properties = {}
    for field_name, field_info in param.annotation.model_fields.items():
      schema.properties[field_name] = _parse_schema_from_parameter(
          variant,
          inspect.Parameter(
              field_name,
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              annotation=field_info.annotation,
          ),
          func_name,
      )
    _raise_if_schema_unsupported(variant, schema)
    return schema
  raise ValueError(
      f'Failed to parse the parameter {param} of function {func_name} for'
      ' automatic function calling. Automatic function calling works best with'
      ' simpler function signature schema,consider manually parse your'
      f' function declaration for function {func_name}.'
  )


def _get_required_fields(schema: types.Schema) -> list[str]:
  if not schema.properties:
    return
  return [
      field_name
      for field_name, field_schema in schema.properties.items()
      if not field_schema.nullable and field_schema.default is None
  ]



================================================
FILE: src/google/adk/tools/function_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any
from typing import Callable
from typing import Optional

from google.genai import types
from typing_extensions import override

from ._automatic_function_calling_util import build_function_declaration
from .base_tool import BaseTool
from .tool_context import ToolContext


class FunctionTool(BaseTool):
  """A tool that wraps a user-defined Python function.

  Attributes:
    func: The function to wrap.
  """

  def __init__(self, func: Callable[..., Any]):
    super().__init__(name=func.__name__, description=func.__doc__)
    self.func = func

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    function_decl = types.FunctionDeclaration.model_validate(
        build_function_declaration(
            func=self.func,
            # The model doesn't understand the function context.
            # input_stream is for streaming tool
            ignore_params=['tool_context', 'input_stream'],
            variant=self._api_variant,
        )
    )

    return function_decl

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if 'tool_context' in signature.parameters:
      args_to_call['tool_context'] = tool_context

    # Before invoking the function, we check for if the list of args passed in
    # has all the mandatory arguments or not.
    # If the check fails, then we don't invoke the tool and let the Agent know
    # that there was a missing a input parameter. This will basically help
    # the underlying model fix the issue and retry.
    mandatory_args = self._get_mandatory_args()
    missing_mandatory_args = [
        arg for arg in mandatory_args if arg not in args_to_call
    ]

    if missing_mandatory_args:
      missing_mandatory_args_str = '\n'.join(missing_mandatory_args)
      error_str = f"""Invoking `{self.name}()` failed as the following mandatory input parameters are not present:
{missing_mandatory_args_str}
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      return {'error': error_str}

    if inspect.iscoroutinefunction(self.func):
      return await self.func(**args_to_call) or {}
    else:
      return self.func(**args_to_call) or {}

  # TODO(hangfei): fix call live for function stream.
  async def _call_live(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
      invocation_context,
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if (
        self.name in invocation_context.active_streaming_tools
        and invocation_context.active_streaming_tools[self.name].stream
    ):
      args_to_call['input_stream'] = invocation_context.active_streaming_tools[
          self.name
      ].stream
    if 'tool_context' in signature.parameters:
      args_to_call['tool_context'] = tool_context
    async for item in self.func(**args_to_call):
      yield item

  def _get_mandatory_args(
      self,
  ) -> list[str]:
    """Identifies mandatory parameters (those without default values) for a function.

    Returns:
      A list of strings, where each string is the name of a mandatory parameter.
    """
    signature = inspect.signature(self.func)
    mandatory_params = []

    for name, param in signature.parameters.items():
      # A parameter is mandatory if:
      # 1. It has no default value (param.default is inspect.Parameter.empty)
      # 2. It's not a variable positional (*args) or variable keyword (**kwargs) parameter
      #
      # For more refer to: https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
      if param.default == inspect.Parameter.empty and param.kind not in (
          inspect.Parameter.VAR_POSITIONAL,
          inspect.Parameter.VAR_KEYWORD,
      ):
        mandatory_params.append(name)

    return mandatory_params



================================================
FILE: src/google/adk/tools/get_user_choice_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
from .long_running_tool import LongRunningFunctionTool
from .tool_context import ToolContext


def get_user_choice(
    options: list[str], tool_context: ToolContext
) -> Optional[str]:
  """Provides the options to the user and asks them to choose one."""
  tool_context.actions.skip_summarization = True
  return None


get_user_choice_tool = LongRunningFunctionTool(func=get_user_choice)



================================================
FILE: src/google/adk/tools/google_search_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class GoogleSearchTool(BaseTool):
  """A built-in tool that is automatically invoked by Gemini 2 models to retrieve search results from Google Search.

  This tool operates internally within the model and does not require or perform
  local code execution.
  """

  def __init__(self):
    # Name and description are not used because this is a model built-in tool.
    super().__init__(name='google_search', description='google_search')

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    llm_request.config = llm_request.config or types.GenerateContentConfig()
    llm_request.config.tools = llm_request.config.tools or []
    if llm_request.model and llm_request.model.startswith('gemini-1'):
      if llm_request.config.tools:
        print(llm_request.config.tools)
        raise ValueError(
            'Google search tool can not be used with other tools in Gemini 1.x.'
        )
      llm_request.config.tools.append(
          types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())
      )
    elif llm_request.model and llm_request.model.startswith('gemini-2'):
      llm_request.config.tools.append(
          types.Tool(google_search=types.GoogleSearch())
      )
    else:
      raise ValueError(
          f'Google search tool is not supported for model {llm_request.model}'
      )


google_search = GoogleSearchTool()



================================================
FILE: src/google/adk/tools/langchain_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from typing import Callable

from google.genai import types
from pydantic import model_validator
from typing_extensions import override

from . import _automatic_function_calling_util
from .function_tool import FunctionTool


class LangchainTool(FunctionTool):
  """Use this class to wrap a langchain tool.

  If the original tool name and description are not suitable, you can override
  them in the constructor.
  """

  tool: Any
  """The wrapped langchain tool."""

  def __init__(self, tool: Any):
    super().__init__(tool._run)
    self.tool = tool
    if tool.name:
      self.name = tool.name
    if tool.description:
      self.description = tool.description

  @model_validator(mode='before')
  @classmethod
  def populate_name(cls, data: Any) -> Any:
    # Override this to not use function's signature name as it's
    # mostly "run" or "invoke" for thir-party tools.
    return data

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    """Build the function declaration for the tool."""
    from langchain.agents import Tool
    from langchain_core.tools import BaseTool

    # There are two types of tools:
    # 1. BaseTool: the tool is defined in langchain.tools.
    # 2. Other tools: the tool doesn't inherit any class but follow some
    #    conventions, like having a "run" method.
    if isinstance(self.tool, BaseTool):
      tool_wrapper = Tool(
          name=self.name,
          func=self.func,
          description=self.description,
      )
      if self.tool.args_schema:
        tool_wrapper.args_schema = self.tool.args_schema
      function_declaration = _automatic_function_calling_util.build_function_declaration_for_langchain(
          False,
          self.name,
          self.description,
          tool_wrapper.func,
          tool_wrapper.args,
      )
      return function_declaration
    else:
      # Need to provide a way to override the function names and descriptions
      # as the original function names are mostly ".run" and the descriptions
      # may not meet users' needs.
      function_declaration = (
          _automatic_function_calling_util.build_function_declaration(
              func=self.tool.run,
          )
      )
      return function_declaration



================================================
FILE: src/google/adk/tools/load_artifacts_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest
  from .tool_context import ToolContext


class LoadArtifactsTool(BaseTool):
  """A tool that loads the artifacts and adds them to the session."""

  def __init__(self):
    super().__init__(
        name='load_artifacts',
        description='Loads the artifacts and adds them to the session.',
    )

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'artifact_names': types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.STRING,
                    ),
                )
            },
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    artifact_names: list[str] = args.get('artifact_names', [])
    return {'artifact_names': artifact_names}

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    await super().process_llm_request(
        tool_context=tool_context,
        llm_request=llm_request,
    )
    await self._append_artifacts_to_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

  async def _append_artifacts_to_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ):
    artifact_names = await tool_context.list_artifacts()
    if not artifact_names:
      return

    # Tell the model about the available artifacts.
    llm_request.append_instructions([f"""You have a list of artifacts:
  {json.dumps(artifact_names)}

  When the user asks questions about any of the artifacts, you should call the
  `load_artifacts` function to load the artifact. Do not generate any text other
  than the function call.
  """])

    # Attach the content of the artifacts if the model requests them.
    # This only adds the content to the model request, instead of the session.
    if llm_request.contents and llm_request.contents[-1].parts:
      function_response = llm_request.contents[-1].parts[0].function_response
      if function_response and function_response.name == 'load_artifacts':
        artifact_names = function_response.response['artifact_names']
        for artifact_name in artifact_names:
          artifact = await tool_context.load_artifact(artifact_name)
          llm_request.contents.append(
              types.Content(
                  role='user',
                  parts=[
                      types.Part.from_text(
                          text=f'Artifact {artifact_name} is:'
                      ),
                      artifact,
                  ],
              )
          )


load_artifacts_tool = LoadArtifactsTool()



================================================
FILE: src/google/adk/tools/load_memory_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from .function_tool import FunctionTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..memory.base_memory_service import MemoryResult
  from ..models import LlmRequest


async def load_memory(
    query: str, tool_context: ToolContext
) -> 'list[MemoryResult]':
  """Loads the memory for the current user.

  Args:
    query: The query to load the memory for.

  Returns:
    A list of memory results.
  """
  response = await tool_context.search_memory(query)
  return response.memories


class LoadMemoryTool(FunctionTool):
  """A tool that loads the memory for the current user."""

  def __init__(self):
    super().__init__(load_memory)

  @override
  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'query': types.Schema(
                    type=types.Type.STRING,
                )
            },
        ),
    )

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    await super().process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )
    # Tell the model about the memory.
    llm_request.append_instructions(["""
You have memory. You can use it to answer questions. If any questions need
you to look up the memory, you should call load_memory function with a query.
"""])


load_memory_tool = LoadMemoryTool()



================================================
FILE: src/google/adk/tools/load_web_page.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool for web browse."""

import requests


def load_web_page(url: str) -> str:
  """Fetches the content in the url and returns the text in it.

  Args:
      url (str): The url to browse.

  Returns:
      str: The text content of the url.
  """
  from bs4 import BeautifulSoup

  response = requests.get(url)

  if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'lxml')
    text = soup.get_text(separator='\n', strip=True)
  else:
    text = f'Failed to fetch url: {url}'

  # Split the text into lines, filtering out very short lines
  # (e.g., single words or short subtitles)
  return '\n'.join(line for line in text.splitlines() if len(line.split()) > 3)



================================================
FILE: src/google/adk/tools/long_running_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

from .function_tool import FunctionTool


class LongRunningFunctionTool(FunctionTool):
  """A function tool that returns the result asynchronously.

  This tool is used for long-running operations that may take a significant
  amount of time to complete. The framework will call the function. Once the
  function returns, the response will be returned asynchronously to the
  framework which is identified by the function_call_id.

  Example:
  ```python
  tool = LongRunningFunctionTool(a_long_running_function)
  ```

  Attributes:
    is_long_running: Whether the tool is a long running operation.
  """

  def __init__(self, func: Callable):
    super().__init__(func)
    self.is_long_running = True



================================================
FILE: src/google/adk/tools/preload_memory_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class PreloadMemoryTool(BaseTool):
  """A tool that preloads the memory for the current user."""

  def __init__(self):
    # Name and description are not used because this tool only
    # changes llm_request.
    super().__init__(name='preload_memory', description='preload_memory')

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    parts = tool_context.user_content.parts
    if not parts or not parts[0].text:
      return
    query = parts[0].text
    response = await tool_context.search_memory(query)
    if not response.memories:
      return
    memory_text = ''
    for memory in response.memories:
      time_str = datetime.fromtimestamp(memory.events[0].timestamp).isoformat()
      memory_text += f'Time: {time_str}\n'
      for event in memory.events:
        # TODO: support multi-part content.
        if (
            event.content
            and event.content.parts
            and event.content.parts[0].text
        ):
          memory_text += f'{event.author}: {event.content.parts[0].text}\n'
    si = f"""The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
{memory_text}
</PAST_CONVERSATIONS>
"""
    llm_request.append_instructions([si])


preload_memory_tool = PreloadMemoryTool()



================================================
FILE: src/google/adk/tools/tool_context.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional
from typing import TYPE_CHECKING

from ..agents.callback_context import CallbackContext
from ..auth.auth_credential import AuthCredential
from ..auth.auth_handler import AuthHandler
from ..auth.auth_tool import AuthConfig

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..events.event_actions import EventActions
  from ..memory.base_memory_service import SearchMemoryResponse


class ToolContext(CallbackContext):
  """The context of the tool.

  This class provides the context for a tool invocation, including access to
  the invocation context, function call ID, event actions, and authentication
  response. It also provides methods for requesting credentials, retrieving
  authentication responses, listing artifacts, and searching memory.

  Attributes:
    invocation_context: The invocation context of the tool.
    function_call_id: The function call id of the current tool call. This id was
      returned in the function call event from LLM to identify a function call.
      If LLM didn't return this id, ADK will assign one to it. This id is used
      to map function call response to the original function call.
    event_actions: The event actions of the current tool call.
  """

  def __init__(
      self,
      invocation_context: InvocationContext,
      *,
      function_call_id: Optional[str] = None,
      event_actions: Optional[EventActions] = None,
  ):
    super().__init__(invocation_context, event_actions=event_actions)
    self.function_call_id = function_call_id

  @property
  def actions(self) -> EventActions:
    return self._event_actions

  def request_credential(self, auth_config: AuthConfig) -> None:
    if not self.function_call_id:
      raise ValueError('function_call_id is not set.')
    self._event_actions.requested_auth_configs[self.function_call_id] = (
        AuthHandler(auth_config).generate_auth_request()
    )

  def get_auth_response(self, auth_config: AuthConfig) -> AuthCredential:
    return AuthHandler(auth_config).get_auth_response(self.state)

  async def list_artifacts(self) -> list[str]:
    """Lists the filenames of the artifacts attached to the current session."""
    if self._invocation_context.artifact_service is None:
      raise ValueError('Artifact service is not initialized.')
    return await self._invocation_context.artifact_service.list_artifact_keys(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        session_id=self._invocation_context.session.id,
    )

  async def search_memory(self, query: str) -> SearchMemoryResponse:
    """Searches the memory of the current user."""
    if self._invocation_context.memory_service is None:
      raise ValueError('Memory service is not available.')
    return await self._invocation_context.memory_service.search_memory(
        app_name=self._invocation_context.app_name,
        user_id=self._invocation_context.user_id,
        query=query,
    )



================================================
FILE: src/google/adk/tools/toolbox_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from . import _automatic_function_calling_util
from .langchain_tool import LangchainTool


class ToolboxTool:
  """A class that provides access to toolbox tools.

  Example:
  ```python
  toolbox = ToolboxTool("http://127.0.0.1:5000")
  tool = toolbox.get_tool("tool_name")
  toolset = toolbox.get_toolset("toolset_name")
  ```
  """

  toolbox_client: Any
  """The toolbox client."""

  def __init__(self, url: str):
    from toolbox_langchain import ToolboxClient

    self.toolbox_client = ToolboxClient(url)

  def get_tool(self, tool_name: str) -> LangchainTool:
    tool = self.toolbox_client.load_tool(tool_name)
    return LangchainTool(tool)

  def get_toolset(self, toolset_name: str) -> list[LangchainTool]:
    tools = self.toolbox_client.load_toolset(toolset_name)
    return [LangchainTool(tool) for tool in tools]



================================================
FILE: src/google/adk/tools/transfer_to_agent_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .tool_context import ToolContext


# TODO: make this internal, since user doesn't need to use this tool directly.
def transfer_to_agent(agent_name: str, tool_context: ToolContext):
  """Transfer the question to another agent."""
  tool_context.actions.transfer_to_agent = agent_name



================================================
FILE: src/google/adk/tools/vertex_ai_search_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models import LlmRequest


class VertexAiSearchTool(BaseTool):
  """A built-in tool using Vertex AI Search.

  Attributes:
    data_store_id: The Vertex AI search data store resource ID.
    search_engine_id: The Vertex AI search engine resource ID.
  """

  def __init__(
      self,
      *,
      data_store_id: Optional[str] = None,
      search_engine_id: Optional[str] = None,
  ):
    """Initializes the Vertex AI Search tool.

    Args:
      data_store_id: The Vertex AI search data store resource ID in the format
        of
        "projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}".
      search_engine_id: The Vertex AI search engine resource ID in the format of
        "projects/{project}/locations/{location}/collections/{collection}/engines/{engine}".

    Raises:
      ValueError: If both data_store_id and search_engine_id are not specified
      or both are specified.
    """
    # Name and description are not used because this is a model built-in tool.
    super().__init__(name='vertex_ai_search', description='vertex_ai_search')
    if (data_store_id is None and search_engine_id is None) or (
        data_store_id is not None and search_engine_id is not None
    ):
      raise ValueError(
          'Either data_store_id or search_engine_id must be specified.'
      )
    self.data_store_id = data_store_id
    self.search_engine_id = search_engine_id

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    if llm_request.model and llm_request.model.startswith('gemini-'):
      if llm_request.model.startswith('gemini-1') and llm_request.config.tools:
        raise ValueError(
            'Vertex AI search tool can not be used with other tools in Gemini'
            ' 1.x.'
        )
      llm_request.config = llm_request.config or types.GenerateContentConfig()
      llm_request.config.tools = llm_request.config.tools or []
      llm_request.config.tools.append(
          types.Tool(
              retrieval=types.Retrieval(
                  vertex_ai_search=types.VertexAISearch(
                      datastore=self.data_store_id, engine=self.search_engine_id
                  )
              )
          )
      )
    else:
      raise ValueError(
          'Vertex AI search tool is not supported for model'
          f' {llm_request.model}'
      )



================================================
FILE: src/google/adk/tools/apihub_tool/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .apihub_toolset import APIHubToolset

__all__ = [
    'APIHubToolset',
]



================================================
FILE: src/google/adk/tools/apihub_tool/apihub_toolset.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, List, Optional

import yaml

from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ..openapi_tool.common.common import to_snake_case
from ..openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from ..openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from .clients.apihub_client import APIHubClient


class APIHubToolset:
  """APIHubTool generates tools from a given API Hub resource.

  Examples:

  ```
  apihub_toolset = APIHubToolset(
      apihub_resource_name="projects/test-project/locations/us-central1/apis/test-api",
      service_account_json="...",
  )

  # Get all available tools
  agent = LlmAgent(tools=apihub_toolset.get_tools())

  # Get a specific tool
  agent = LlmAgent(tools=[
      ...
      apihub_toolset.get_tool('my_tool'),
  ])
  ```

  **apihub_resource_name** is the resource name from API Hub. It must include
    API name, and can optionally include API version and spec name.
    - If apihub_resource_name includes a spec resource name, the content of that
      spec will be used for generating the tools.
    - If apihub_resource_name includes only an api or a version name, the
      first spec of the first version of that API will be used.
  """

  def __init__(
      self,
      *,
      # Parameters for fetching API Hub resource
      apihub_resource_name: str,
      access_token: Optional[str] = None,
      service_account_json: Optional[str] = None,
      # Parameters for the toolset itself
      name: str = '',
      description: str = '',
      # Parameters for generating tools
      lazy_load_spec=False,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
      # Optionally, you can provide a custom API Hub client
      apihub_client: Optional[APIHubClient] = None,
  ):
    """Initializes the APIHubTool with the given parameters.

    Examples:
    ```
    apihub_toolset = APIHubToolset(
        apihub_resource_name="projects/test-project/locations/us-central1/apis/test-api",
        service_account_json="...",
    )

    # Get all available tools
    agent = LlmAgent(tools=apihub_toolset.get_tools())

    # Get a specific tool
    agent = LlmAgent(tools=[
        ...
        apihub_toolset.get_tool('my_tool'),
    ])
    ```

    **apihub_resource_name** is the resource name from API Hub. It must include
    API name, and can optionally include API version and spec name.
    - If apihub_resource_name includes a spec resource name, the content of that
      spec will be used for generating the tools.
    - If apihub_resource_name includes only an api or a version name, the
      first spec of the first version of that API will be used.

    Example:
    * projects/xxx/locations/us-central1/apis/apiname/...
    * https://console.cloud.google.com/apigee/api-hub/apis/apiname?project=xxx

    Args:
        apihub_resource_name: The resource name of the API in API Hub.
          Example: `projects/test-project/locations/us-central1/apis/test-api`.
        access_token: Google Access token. Generate with gcloud cli `gcloud auth
          auth print-access-token`. Used for fetching API Specs from API Hub.
        service_account_json: The service account config as a json string.
          Required if not using default service credential. It is used for
          creating the API Hub client and fetching the API Specs from API Hub.
        apihub_client: Optional custom API Hub client.
        name: Name of the toolset. Optional.
        description: Description of the toolset. Optional.
        auth_scheme: Auth scheme that applies to all the tool in the toolset.
        auth_credential: Auth credential that applies to all the tool in the
          toolset.
        lazy_load_spec: If True, the spec will be loaded lazily when needed.
          Otherwise, the spec will be loaded immediately and the tools will be
          generated during initialization.
    """
    self.name = name
    self.description = description
    self.apihub_resource_name = apihub_resource_name
    self.lazy_load_spec = lazy_load_spec
    self.apihub_client = apihub_client or APIHubClient(
        access_token=access_token,
        service_account_json=service_account_json,
    )

    self.generated_tools: Dict[str, RestApiTool] = {}
    self.auth_scheme = auth_scheme
    self.auth_credential = auth_credential

    if not self.lazy_load_spec:
      self._prepare_tools()

  def get_tool(self, name: str) -> Optional[RestApiTool]:
    """Retrieves a specific tool by its name.

    Example:
    ```
    apihub_tool = apihub_toolset.get_tool('my_tool')
    ```

    Args:
        name: The name of the tool to retrieve.

    Returns:
        The tool with the given name, or None if no such tool exists.
    """
    if not self._are_tools_ready():
      self._prepare_tools()

    return self.generated_tools[name] if name in self.generated_tools else None

  def get_tools(self) -> List[RestApiTool]:
    """Retrieves all available tools.

    Returns:
        A list of all available RestApiTool objects.
    """
    if not self._are_tools_ready():
      self._prepare_tools()

    return list(self.generated_tools.values())

  def _are_tools_ready(self) -> bool:
    return not self.lazy_load_spec or self.generated_tools

  def _prepare_tools(self) -> str:
    """Fetches the spec from API Hub and generates the tools.

    Returns:
        True if the tools are ready, False otherwise.
    """
    # For each API, get the first version and the first spec of that version.
    spec = self.apihub_client.get_spec_content(self.apihub_resource_name)
    self.generated_tools: Dict[str, RestApiTool] = {}

    tools = self._parse_spec_to_tools(spec)
    for tool in tools:
      self.generated_tools[tool.name] = tool

  def _parse_spec_to_tools(self, spec_str: str) -> List[RestApiTool]:
    """Parses the spec string to a list of RestApiTool.

    Args:
        spec_str: The spec string to parse.

    Returns:
        A list of RestApiTool objects.
    """
    spec_dict = yaml.safe_load(spec_str)
    if not spec_dict:
      return []

    self.name = self.name or to_snake_case(
        spec_dict.get('info', {}).get('title', 'unnamed')
    )
    self.description = self.description or spec_dict.get('info', {}).get(
        'description', ''
    )
    tools = OpenAPIToolset(
        spec_dict=spec_dict,
        auth_credential=self.auth_credential,
        auth_scheme=self.auth_scheme,
    ).get_tools()
    return tools



================================================
FILE: src/google/adk/tools/apihub_tool/clients/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



================================================
FILE: src/google/adk/tools/apihub_tool/clients/apihub_client.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
import base64
import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
from google.auth import default as default_service_credential
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import requests


class BaseAPIHubClient(ABC):
  """Base class for API Hub clients."""

  @abstractmethod
  def get_spec_content(self, resource_name: str) -> str:
    """From a given resource name, get the soec in the API Hub."""
    raise NotImplementedError()


class APIHubClient(BaseAPIHubClient):
  """Client for interacting with the API Hub service."""

  def __init__(
      self,
      *,
      access_token: Optional[str] = None,
      service_account_json: Optional[str] = None,
  ):
    """Initializes the APIHubClient.

    You must set either access_token or service_account_json. This
    credential is used for sending request to API Hub API.

    Args:
        access_token: Google Access token. Generate with gcloud cli `gcloud auth
          print-access-token`. Useful for local testing.
        service_account_json: The service account configuration as a dictionary.
          Required if not using default service credential.
    """
    self.root_url = "https://apihub.googleapis.com/v1"
    self.credential_cache = None
    self.access_token, self.service_account = None, None

    if access_token:
      self.access_token = access_token
    elif service_account_json:
      self.service_account = service_account_json

  def get_spec_content(self, path: str) -> str:
    """From a given path, get the first spec available in the API Hub.

    - If path includes /apis/apiname, get the first spec of that API
    - If path includes /apis/apiname/versions/versionname, get the first spec
      of that API Version
    - If path includes /apis/apiname/versions/versionname/specs/specname, return
      that spec

    Path can be resource name (projects/xxx/locations/us-central1/apis/apiname),
    and URL from the UI
    (https://console.cloud.google.com/apigee/api-hub/apis/apiname?project=xxx)

    Args:
        path: The path to the API, API Version, or API Spec.

    Returns:
        The content of the first spec available in the API Hub.
    """
    apihub_resource_name, api_version_resource_name, api_spec_resource_name = (
        self._extract_resource_name(path)
    )

    if apihub_resource_name and not api_version_resource_name:
      api = self.get_api(apihub_resource_name)
      versions = api.get("versions", [])
      if not versions:
        raise ValueError(
            f"No versions found in API Hub resource: {apihub_resource_name}"
        )
      api_version_resource_name = versions[0]

    if api_version_resource_name and not api_spec_resource_name:
      api_version = self.get_api_version(api_version_resource_name)
      spec_resource_names = api_version.get("specs", [])
      if not spec_resource_names:
        raise ValueError(
            f"No specs found in API Hub version: {api_version_resource_name}"
        )
      api_spec_resource_name = spec_resource_names[0]

    if api_spec_resource_name:
      spec_content = self._fetch_spec(api_spec_resource_name)
      return spec_content

    raise ValueError("No API Hub resource found in path: {path}")

  def list_apis(self, project: str, location: str) -> List[Dict[str, Any]]:
    """Lists all APIs in the specified project and location.

    Args:
        project: The Google Cloud project name.
        location: The location of the API Hub resources (e.g., 'us-central1').

    Returns:
        A list of API dictionaries, or an empty list if an error occurs.
    """
    url = f"{self.root_url}/projects/{project}/locations/{location}/apis"
    headers = {
        "accept": "application/json, text/plain, */*",
        "Authorization": f"Bearer {self._get_access_token()}",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    apis = response.json().get("apis", [])
    return apis

  def get_api(self, api_resource_name: str) -> Dict[str, Any]:
    """Get API detail by API name.

    Args:
        api_resource_name: Resource name of this API, like
          projects/xxx/locations/us-central1/apis/apiname

    Returns:
        An API and details in a dict.
    """
    url = f"{self.root_url}/{api_resource_name}"
    headers = {
        "accept": "application/json, text/plain, */*",
        "Authorization": f"Bearer {self._get_access_token()}",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    apis = response.json()
    return apis

  def get_api_version(self, api_version_name: str) -> Dict[str, Any]:
    """Gets details of a specific API version.

    Args:
        api_version_name: The resource name of the API version.

    Returns:
        The API version details as a dictionary, or an empty dictionary if an
        error occurs.
    """
    url = f"{self.root_url}/{api_version_name}"
    headers = {
        "accept": "application/json, text/plain, */*",
        "Authorization": f"Bearer {self._get_access_token()}",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

  def _fetch_spec(self, api_spec_resource_name: str) -> str:
    """Retrieves the content of a specific API specification.

    Args:
        api_spec_resource_name: The resource name of the API spec.

    Returns:
        The decoded content of the specification as a string, or an empty string
        if an error occurs.
    """
    url = f"{self.root_url}/{api_spec_resource_name}:contents"
    headers = {
        "accept": "application/json, text/plain, */*",
        "Authorization": f"Bearer {self._get_access_token()}",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content_base64 = response.json().get("contents", "")
    if content_base64:
      content_decoded = base64.b64decode(content_base64).decode("utf-8")
      return content_decoded
    else:
      return ""

  def _extract_resource_name(self, url_or_path: str) -> Tuple[str, str, str]:
    """Extracts the resource names of an API, API Version, and API Spec from a given URL or path.

    Args:
        url_or_path: The URL (UI or resource) or path string.

    Returns:
        A dictionary containing the resource names:
        {
            "api_resource_name": "projects/*/locations/*/apis/*",
            "api_version_resource_name":
            "projects/*/locations/*/apis/*/versions/*",
            "api_spec_resource_name":
            "projects/*/locations/*/apis/*/versions/*/specs/*"
        }
        or raises ValueError if extraction fails.

    Raises:
        ValueError: If the URL or path is invalid or if required components
        (project, location, api) are missing.
    """

    query_params = None
    try:
      parsed_url = urlparse(url_or_path)
      path = parsed_url.path
      query_params = parse_qs(parsed_url.query)

      # This is a path from UI. Remove unnecessary prefix.
      if "api-hub/" in path:
        path = path.split("api-hub")[1]
    except Exception:
      path = url_or_path

    path_segments = [segment for segment in path.split("/") if segment]

    project = None
    location = None
    api_id = None
    version_id = None
    spec_id = None

    if "projects" in path_segments:
      project_index = path_segments.index("projects")
      if project_index + 1 < len(path_segments):
        project = path_segments[project_index + 1]
    elif query_params and "project" in query_params:
      project = query_params["project"][0]

    if not project:
      raise ValueError(
          "Project ID not found in URL or path in APIHubClient. Input path is"
          f" '{url_or_path}'. Please make sure there is either"
          " '/projects/PROJECT_ID' in the path or 'project=PROJECT_ID' query"
          " param in the input."
      )

    if "locations" in path_segments:
      location_index = path_segments.index("locations")
      if location_index + 1 < len(path_segments):
        location = path_segments[location_index + 1]
    if not location:
      raise ValueError(
          "Location not found in URL or path in APIHubClient. Input path is"
          f" '{url_or_path}'. Please make sure there is either"
          " '/location/LOCATION_ID' in the path."
      )

    if "apis" in path_segments:
      api_index = path_segments.index("apis")
      if api_index + 1 < len(path_segments):
        api_id = path_segments[api_index + 1]
    if not api_id:
      raise ValueError(
          "API id not found in URL or path in APIHubClient. Input path is"
          f" '{url_or_path}'. Please make sure there is either"
          " '/apis/API_ID' in the path."
      )
    if "versions" in path_segments:
      version_index = path_segments.index("versions")
      if version_index + 1 < len(path_segments):
        version_id = path_segments[version_index + 1]

    if "specs" in path_segments:
      spec_index = path_segments.index("specs")
      if spec_index + 1 < len(path_segments):
        spec_id = path_segments[spec_index + 1]

    api_resource_name = f"projects/{project}/locations/{location}/apis/{api_id}"
    api_version_resource_name = (
        f"{api_resource_name}/versions/{version_id}" if version_id else None
    )
    api_spec_resource_name = (
        f"{api_version_resource_name}/specs/{spec_id}"
        if version_id and spec_id
        else None
    )

    return (
        api_resource_name,
        api_version_resource_name,
        api_spec_resource_name,
    )

  def _get_access_token(self) -> str:
    """Gets the access token for the service account.

    Returns:
        The access token.
    """
    if self.access_token:
      return self.access_token

    if self.credential_cache and not self.credential_cache.expired:
      return self.credential_cache.token

    if self.service_account:
      try:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(self.service_account),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid service account JSON: {e}") from e
    else:
      try:
        credentials, _ = default_service_credential()
      except:
        credentials = None

    if not credentials:
      raise ValueError(
          "Please provide a service account or an access token to API Hub"
          " client."
      )

    credentials.refresh(Request())
    self.credential_cache = credentials
    return credentials.token



================================================
FILE: src/google/adk/tools/apihub_tool/clients/secret_client.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Optional
import google.auth
from google.auth import default as default_service_credential
import google.auth.transport.requests
from google.cloud import secretmanager
from google.oauth2 import service_account


class SecretManagerClient:
  """A client for interacting with Google Cloud Secret Manager.

  This class provides a simplified interface for retrieving secrets from
  Secret Manager, handling authentication using either a service account
  JSON keyfile (passed as a string) or a pre-existing authorization token.

  Attributes:
      _credentials:  Google Cloud credentials object (ServiceAccountCredentials
        or Credentials).
      _client: Secret Manager client instance.
  """

  def __init__(
      self,
      service_account_json: Optional[str] = None,
      auth_token: Optional[str] = None,
  ):
    """Initializes the SecretManagerClient.

    Args:
        service_account_json:  The content of a service account JSON keyfile (as
          a string), not the file path.  Must be valid JSON.
        auth_token: An existing Google Cloud authorization token.

    Raises:
        ValueError: If neither `service_account_json` nor `auth_token` is
        provided,
            or if both are provided.  Also raised if the service_account_json
            is not valid JSON.
        google.auth.exceptions.GoogleAuthError: If authentication fails.
    """
    if service_account_json:
      try:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(service_account_json)
        )
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid service account JSON: {e}") from e
    elif auth_token:
      credentials = google.auth.credentials.Credentials(
          token=auth_token,
          refresh_token=None,
          token_uri=None,
          client_id=None,
          client_secret=None,
      )
      request = google.auth.transport.requests.Request()
      credentials.refresh(request)
    else:
      try:
        credentials, _ = default_service_credential()
      except Exception as e:
        raise ValueError(
            "'service_account_json' or 'auth_token' are both missing, and"
            f" error occurred while trying to use default credentials: {e}"
        ) from e

    if not credentials:
      raise ValueError(
          "Must provide either 'service_account_json' or 'auth_token', not both"
          " or neither."
      )

    self._credentials = credentials
    self._client = secretmanager.SecretManagerServiceClient(
        credentials=self._credentials
    )

  def get_secret(self, resource_name: str) -> str:
    """Retrieves a secret from Google Cloud Secret Manager.

    Args:
        resource_name: The full resource name of the secret, in the format
          "projects/*/secrets/*/versions/*".  Usually you want the "latest"
          version, e.g.,
          "projects/my-project/secrets/my-secret/versions/latest".

    Returns:
        The secret payload as a string.

    Raises:
        google.api_core.exceptions.GoogleAPIError: If the Secret Manager API
            returns an error (e.g., secret not found, permission denied).
        Exception: For other unexpected errors.
    """
    try:
      response = self._client.access_secret_version(name=resource_name)
      return response.payload.data.decode("UTF-8")
    except Exception as e:
      raise e  # Re-raise the exception to allow for handling by the caller
      # Consider logging the exception here before re-raising.



================================================
FILE: src/google/adk/tools/application_integration_tool/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .application_integration_toolset import ApplicationIntegrationToolset
from .integration_connector_tool import IntegrationConnectorTool

__all__ = [
    'ApplicationIntegrationToolset',
    'IntegrationConnectorTool',
]



================================================
FILE: src/google/adk/tools/application_integration_tool/application_integration_toolset.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

from fastapi.openapi.models import HTTPBearer

from ...auth.auth_credential import AuthCredential
from ...auth.auth_credential import AuthCredentialTypes
from ...auth.auth_credential import ServiceAccount
from ...auth.auth_credential import ServiceAccountCredential
from ..openapi_tool.auth.auth_helpers import service_account_scheme_credential
from ..openapi_tool.openapi_spec_parser.openapi_spec_parser import OpenApiSpecParser
from ..openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from ..openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from .clients.connections_client import ConnectionsClient
from .clients.integration_client import IntegrationClient
from .integration_connector_tool import IntegrationConnectorTool


# TODO(cheliu): Apply a common toolset interface
class ApplicationIntegrationToolset:
  """ApplicationIntegrationToolset generates tools from a given Application

  Integration or Integration Connector resource.
  Example Usage:
  ```
  # Get all available tools for an integration with api trigger
  application_integration_toolset = ApplicationIntegrationToolset(

      project="test-project",
      location="us-central1"
      integration="test-integration",
      trigger="api_trigger/test_trigger",
      service_account_credentials={...},
  )

  # Get all available tools for a connection using entity operations and
  # actions
  # Note: Find the list of supported entity operations and actions for a
  connection
  # using integration connector apis:
  #
  https://cloud.google.com/integration-connectors/docs/reference/rest/v1/projects.locations.connections.connectionSchemaMetadata
  application_integration_toolset = ApplicationIntegrationToolset(
      project="test-project",
      location="us-central1"
      connection="test-connection",
      entity_operations=["EntityId1": ["LIST","CREATE"], "EntityId2": []],
      #empty list for actions means all operations on the entity are supported
      actions=["action1"],
      service_account_credentials={...},
  )

  # Get all available tools
  agent = LlmAgent(tools=[
      ...
      *application_integration_toolset.get_tools(),
  ])
  ```
  """

  def __init__(
      self,
      project: str,
      location: str,
      integration: Optional[str] = None,
      triggers: Optional[List[str]] = None,
      connection: Optional[str] = None,
      entity_operations: Optional[str] = None,
      actions: Optional[str] = None,
      # Optional parameter for the toolset. This is prepended to the generated
      # tool/python function name.
      tool_name: Optional[str] = "",
      # Optional parameter for the toolset. This is appended to the generated
      # tool/python function description.
      tool_instructions: Optional[str] = "",
      service_account_json: Optional[str] = None,
  ):
    """Initializes the ApplicationIntegrationToolset.

    Example Usage:
    ```
    # Get all available tools for an integration with api trigger
    application_integration_toolset = ApplicationIntegrationToolset(

        project="test-project",
        location="us-central1"
        integration="test-integration",
        triggers=["api_trigger/test_trigger"],
        service_account_credentials={...},
    )

    # Get all available tools for a connection using entity operations and
    # actions
    # Note: Find the list of supported entity operations and actions for a
    connection
    # using integration connector apis:
    #
    https://cloud.google.com/integration-connectors/docs/reference/rest/v1/projects.locations.connections.connectionSchemaMetadata
    application_integration_toolset = ApplicationIntegrationToolset(
        project="test-project",
        location="us-central1"
        connection="test-connection",
        entity_operations=["EntityId1": ["LIST","CREATE"], "EntityId2": []],
        #empty list for actions means all operations on the entity are supported
        actions=["action1"],
        service_account_credentials={...},
    )

    # Get all available tools
    agent = LlmAgent(tools=[
        ...
        *application_integration_toolset.get_tools(),
    ])
    ```

    Args:
        project: The GCP project ID.
        location: The GCP location.
        integration: The integration name.
        triggers: The list of trigger names in the integration.
        connection: The connection name.
        entity_operations: The entity operations supported by the connection.
        actions: The actions supported by the connection.
        tool_name: The name of the tool.
        tool_instructions: The instructions for the tool.
        service_account_json: The service account configuration as a dictionary.
          Required if not using default service credential. Used for fetching
          the Application Integration or Integration Connector resource.

    Raises:
        ValueError: If neither integration and trigger nor connection and
            (entity_operations or actions) is provided.
        Exception: If there is an error during the initialization of the
            integration or connection client.
    """
    self.project = project
    self.location = location
    self.integration = integration
    self.triggers = triggers
    self.connection = connection
    self.entity_operations = entity_operations
    self.actions = actions
    self.tool_name = tool_name
    self.tool_instructions = tool_instructions
    self.service_account_json = service_account_json
    self.generated_tools: Dict[str, RestApiTool] = {}

    integration_client = IntegrationClient(
        project,
        location,
        integration,
        triggers,
        connection,
        entity_operations,
        actions,
        service_account_json,
    )
    connection_details = {}
    if integration:
      spec = integration_client.get_openapi_spec_for_integration()
    elif connection and (entity_operations or actions):
      connections_client = ConnectionsClient(
          project, location, connection, service_account_json
      )
      connection_details = connections_client.get_connection_details()
      spec = integration_client.get_openapi_spec_for_connection(
          tool_name,
          tool_instructions,
      )
    else:
      raise ValueError(
          "Either (integration and trigger) or (connection and"
          " (entity_operations or actions)) should be provided."
      )
    self._parse_spec_to_tools(spec, connection_details)

  def _parse_spec_to_tools(self, spec_dict, connection_details):
    """Parses the spec dict to a list of RestApiTool."""
    if self.service_account_json:
      sa_credential = ServiceAccountCredential.model_validate_json(
          self.service_account_json
      )
      service_account = ServiceAccount(
          service_account_credential=sa_credential,
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
      auth_scheme, auth_credential = service_account_scheme_credential(
          config=service_account
      )
    else:
      auth_credential = AuthCredential(
          auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
          service_account=ServiceAccount(
              use_default_credential=True,
              scopes=["https://www.googleapis.com/auth/cloud-platform"],
          ),
      )
      auth_scheme = HTTPBearer(bearerFormat="JWT")

    if self.integration:
      tools = OpenAPIToolset(
          spec_dict=spec_dict,
          auth_credential=auth_credential,
          auth_scheme=auth_scheme,
      ).get_tools()
      for tool in tools:
        self.generated_tools[tool.name] = tool
      return

    operations = OpenApiSpecParser().parse(spec_dict)

    for open_api_operation in operations:
      operation = getattr(open_api_operation.operation, "x-operation")
      entity = None
      action = None
      if hasattr(open_api_operation.operation, "x-entity"):
        entity = getattr(open_api_operation.operation, "x-entity")
      elif hasattr(open_api_operation.operation, "x-action"):
        action = getattr(open_api_operation.operation, "x-action")
      rest_api_tool = RestApiTool.from_parsed_operation(open_api_operation)
      if auth_scheme:
        rest_api_tool.configure_auth_scheme(auth_scheme)
      if auth_credential:
        rest_api_tool.configure_auth_credential(auth_credential)
      tool = IntegrationConnectorTool(
          name=rest_api_tool.name,
          description=rest_api_tool.description,
          connection_name=connection_details["name"],
          connection_host=connection_details["host"],
          connection_service_name=connection_details["serviceName"],
          entity=entity,
          action=action,
          operation=operation,
          rest_api_tool=rest_api_tool,
      )
      self.generated_tools[tool.name] = tool

  def get_tools(self) -> List[RestApiTool]:
    return list(self.generated_tools.values())



================================================
FILE: src/google/adk/tools/application_integration_tool/integration_connector_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from typing import Any
from typing import Dict
from typing import Optional

from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import to_gemini_schema
from google.genai.types import FunctionDeclaration
from typing_extensions import override

from .. import BaseTool
from ..tool_context import ToolContext

logger = logging.getLogger(__name__)


class IntegrationConnectorTool(BaseTool):
  """A tool that wraps a RestApiTool to interact with a specific Application Integration endpoint.

  This tool adds Application Integration specific context like connection
  details, entity, operation, and action to the underlying REST API call
  handled by RestApiTool. It prepares the arguments and then delegates the
  actual API call execution to the contained RestApiTool instance.

  * Generates request params and body
  * Attaches auth credentials to API call.

  Example:
  ```
    # Each API operation in the spec will be turned into its own tool
    # Name of the tool is the operationId of that operation, in snake case
    operations = OperationGenerator().parse(openapi_spec_dict)
    tool = [RestApiTool.from_parsed_operation(o) for o in operations]
  ```
  """

  EXCLUDE_FIELDS = [
      'connection_name',
      'service_name',
      'host',
      'entity',
      'operation',
      'action',
  ]

  OPTIONAL_FIELDS = [
      'page_size',
      'page_token',
      'filter',
  ]

  def __init__(
      self,
      name: str,
      description: str,
      connection_name: str,
      connection_host: str,
      connection_service_name: str,
      entity: str,
      operation: str,
      action: str,
      rest_api_tool: RestApiTool,
  ):
    """Initializes the ApplicationIntegrationTool.

    Args:
        name: The name of the tool, typically derived from the API operation.
          Should be unique and adhere to Gemini function naming conventions
          (e.g., less than 64 characters).
        description: A description of what the tool does, usually based on the
          API operation's summary or description.
        connection_name: The name of the Integration Connector connection.
        connection_host: The hostname or IP address for the connection.
        connection_service_name: The specific service name within the host.
        entity: The Integration Connector entity being targeted.
        operation: The specific operation being performed on the entity.
        action: The action associated with the operation (e.g., 'execute').
        rest_api_tool: An initialized RestApiTool instance that handles the
          underlying REST API communication based on an OpenAPI specification
          operation. This tool will be called by ApplicationIntegrationTool with
          added connection and context arguments. tool =
          [RestApiTool.from_parsed_operation(o) for o in operations]
    """
    # Gemini restrict the length of function name to be less than 64 characters
    super().__init__(
        name=name,
        description=description,
    )
    self.connection_name = connection_name
    self.connection_host = connection_host
    self.connection_service_name = connection_service_name
    self.entity = entity
    self.operation = operation
    self.action = action
    self.rest_api_tool = rest_api_tool

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Returns the function declaration in the Gemini Schema format."""
    schema_dict = self.rest_api_tool._operation_parser.get_json_schema()
    for field in self.EXCLUDE_FIELDS:
      if field in schema_dict['properties']:
        del schema_dict['properties'][field]
    for field in self.OPTIONAL_FIELDS + self.EXCLUDE_FIELDS:
      if field in schema_dict['required']:
        schema_dict['required'].remove(field)

    parameters = to_gemini_schema(schema_dict)
    function_decl = FunctionDeclaration(
        name=self.name, description=self.description, parameters=parameters
    )
    return function_decl

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: Optional[ToolContext]
  ) -> Dict[str, Any]:
    args['connection_name'] = self.connection_name
    args['service_name'] = self.connection_service_name
    args['host'] = self.connection_host
    args['entity'] = self.entity
    args['operation'] = self.operation
    args['action'] = self.action
    logger.info('Running tool: %s with args: %s', self.name, args)
    return self.rest_api_tool.call(args=args, tool_context=tool_context)

  def __str__(self):
    return (
        f'ApplicationIntegrationTool(name="{self.name}",'
        f' description="{self.description}",'
        f' connection_name="{self.connection_name}", entity="{self.entity}",'
        f' operation="{self.operation}", action="{self.action}")'
    )

  def __repr__(self):
    return (
        f'ApplicationIntegrationTool(name="{self.name}",'
        f' description="{self.description}",'
        f' connection_name="{self.connection_name}",'
        f' connection_host="{self.connection_host}",'
        f' connection_service_name="{self.connection_service_name}",'
        f' entity="{self.entity}", operation="{self.operation}",'
        f' action="{self.action}", rest_api_tool={repr(self.rest_api_tool)})'
    )



================================================
FILE: src/google/adk/tools/application_integration_tool/clients/connections_client.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import google.auth
from google.auth import default as default_service_credential
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import requests


class ConnectionsClient:
  """Utility class for interacting with Google Cloud Connectors API."""

  def __init__(
      self,
      project: str,
      location: str,
      connection: str,
      service_account_json: Optional[str] = None,
  ):
    """Initializes the ConnectionsClient.

    Args:
      project: The Google Cloud project ID.
      location: The Google Cloud location (e.g., us-central1).
      connection: The connection name.
      service_account_json: The service account configuration as a dictionary.
        Required if not using default service credential. Used for fetching
        connection details.
    """
    self.project = project
    self.location = location
    self.connection = connection
    self.connector_url = "https://connectors.googleapis.com"
    self.service_account_json = service_account_json
    self.credential_cache = None

  def get_connection_details(self) -> Dict[str, Any]:
    """Retrieves service details (service name and host) for a given connection.

    Also returns if auth override is enabled for the connection.

    Returns:
        tuple: A tuple containing (service_name, host).

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request error.
        Exception: For any other unexpected errors.
    """
    url = f"{self.connector_url}/v1/projects/{self.project}/locations/{self.location}/connections/{self.connection}?view=BASIC"

    response = self._execute_api_call(url)

    connection_data = response.json()
    connection_name = connection_data.get("name", "")
    service_name = connection_data.get("serviceDirectory", "")
    host = connection_data.get("host", "")
    if host:
      service_name = connection_data.get("tlsServiceDirectory", "")
    auth_override_enabled = connection_data.get("authOverrideEnabled", False)
    return {
        "name": connection_name,
        "serviceName": service_name,
        "host": host,
        "authOverrideEnabled": auth_override_enabled,
    }

  def get_entity_schema_and_operations(
      self, entity: str
  ) -> Tuple[Dict[str, Any], List[str]]:
    """Retrieves the JSON schema for a given entity in a connection.

    Args:
        entity (str): The entity name.

    Returns:
        tuple: A tuple containing (schema, operations).

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request or processing error.
        Exception: For any other unexpected errors.
    """
    url = f"{self.connector_url}/v1/projects/{self.project}/locations/{self.location}/connections/{self.connection}/connectionSchemaMetadata:getEntityType?entityId={entity}"

    response = self._execute_api_call(url)
    operation_id = response.json().get("name")

    if not operation_id:
      raise ValueError(
          f"Failed to get entity schema and operations for entity: {entity}"
      )

    operation_response = self._poll_operation(operation_id)

    schema = operation_response.get("response", {}).get("jsonSchema", {})
    operations = operation_response.get("response", {}).get("operations", [])
    return schema, operations

  def get_action_schema(self, action: str) -> Dict[str, Any]:
    """Retrieves the input and output JSON schema for a given action in a connection.

    Args:
        action (str): The action name.

    Returns:
        tuple: A tuple containing (input_schema, output_schema).

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request or processing error.
        Exception: For any other unexpected errors.
    """
    url = f"{self.connector_url}/v1/projects/{self.project}/locations/{self.location}/connections/{self.connection}/connectionSchemaMetadata:getAction?actionId={action}"

    response = self._execute_api_call(url)

    operation_id = response.json().get("name")

    if not operation_id:
      raise ValueError(f"Failed to get action schema for action: {action}")

    operation_response = self._poll_operation(operation_id)

    input_schema = operation_response.get("response", {}).get(
        "inputJsonSchema", {}
    )
    output_schema = operation_response.get("response", {}).get(
        "outputJsonSchema", {}
    )
    description = operation_response.get("response", {}).get("description", "")
    display_name = operation_response.get("response", {}).get("displayName", "")
    return {
        "inputSchema": input_schema,
        "outputSchema": output_schema,
        "description": description,
        "displayName": display_name,
    }

  @staticmethod
  def get_connector_base_spec() -> Dict[str, Any]:
    return {
        "openapi": "3.0.1",
        "info": {
            "title": "ExecuteConnection",
            "description": "This tool can execute a query on connection",
            "version": "4",
        },
        "servers": [{"url": "https://integrations.googleapis.com"}],
        "security": [
            {"google_auth": ["https://www.googleapis.com/auth/cloud-platform"]}
        ],
        "paths": {},
        "components": {
            "schemas": {
                "operation": {
                    "type": "string",
                    "default": "LIST_ENTITIES",
                    "description": (
                        "Operation to execute. Possible values are"
                        " LIST_ENTITIES, GET_ENTITY, CREATE_ENTITY,"
                        " UPDATE_ENTITY, DELETE_ENTITY in case of entities."
                        " EXECUTE_ACTION in case of actions. and EXECUTE_QUERY"
                        " in case of custom queries."
                    ),
                },
                "entityId": {
                    "type": "string",
                    "description": "Name of the entity",
                },
                "connectorInputPayload": {"type": "object"},
                "filterClause": {
                    "type": "string",
                    "default": "",
                    "description": "WHERE clause in SQL query",
                },
                "pageSize": {
                    "type": "integer",
                    "default": 50,
                    "description": (
                        "Number of entities to return in the response"
                    ),
                },
                "pageToken": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Page token to return the next page of entities"
                    ),
                },
                "connectionName": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Connection resource name to run the query for"
                    ),
                },
                "serviceName": {
                    "type": "string",
                    "default": "",
                    "description": "Service directory for the connection",
                },
                "host": {
                    "type": "string",
                    "default": "",
                    "description": "Host name incase of tls service directory",
                },
                "entity": {
                    "type": "string",
                    "default": "Issues",
                    "description": "Entity to run the query for",
                },
                "action": {
                    "type": "string",
                    "default": "ExecuteCustomQuery",
                    "description": "Action to run the query for",
                },
                "query": {
                    "type": "string",
                    "default": "",
                    "description": "Custom Query to execute on the connection",
                },
                "dynamicAuthConfig": {
                    "type": "object",
                    "default": {},
                    "description": "Dynamic auth config for the connection",
                },
                "timeout": {
                    "type": "integer",
                    "default": 120,
                    "description": (
                        "Timeout in seconds for execution of custom query"
                    ),
                },
                "connectorOutputPayload": {"type": "object"},
                "nextPageToken": {"type": "string"},
                "execute-connector_Response": {
                    "required": ["connectorOutputPayload"],
                    "type": "object",
                    "properties": {
                        "connectorOutputPayload": {
                            "$ref": (
                                "#/components/schemas/connectorOutputPayload"
                            )
                        },
                        "nextPageToken": {
                            "$ref": "#/components/schemas/nextPageToken"
                        },
                    },
                },
            },
            "securitySchemes": {
                "google_auth": {
                    "type": "oauth2",
                    "flows": {
                        "implicit": {
                            "authorizationUrl": (
                                "https://accounts.google.com/o/oauth2/auth"
                            ),
                            "scopes": {
                                "https://www.googleapis.com/auth/cloud-platform": (
                                    "Auth for google cloud services"
                                )
                            },
                        }
                    },
                }
            },
        },
    }

  @staticmethod
  def get_action_operation(
      action: str,
      operation: str,
      action_display_name: str,
      tool_name: str = "",
      tool_instructions: str = "",
  ) -> Dict[str, Any]:
    description = f"Use this tool to execute {action}"
    if operation == "EXECUTE_QUERY":
      description += (
          " Use pageSize = 50 and timeout = 120 until user specifies a"
          " different value otherwise. If user provides a query in natural"
          " language, convert it to SQL query and then execute it using the"
          " tool."
      )
    return {
        "post": {
            "summary": f"{action_display_name}",
            "description": f"{description} {tool_instructions}",
            "operationId": f"{tool_name}_{action_display_name}",
            "x-action": f"{action}",
            "x-operation": f"{operation}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": f"#/components/schemas/{action_display_name}_Request"
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": f"#/components/schemas/{action_display_name}_Response",
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def list_operation(
      entity: str,
      schema_as_string: str = "",
      tool_name: str = "",
      tool_instructions: str = "",
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"List {entity}",
            "description": f"""Returns the list of {entity} data. If the page token was available in the response, let users know there are more records available. Ask if the user wants to fetch the next page of results. When passing filter use the
                following format: `field_name1='value1' AND field_name2='value2'
                `. {tool_instructions}""",
            "x-operation": "LIST_ENTITIES",
            "x-entity": f"{entity}",
            "operationId": f"{tool_name}_list_{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": (
                                f"#/components/schemas/list_{entity}_Request"
                            )
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "description": (
                                    f"Returns a list of {entity} of json"
                                    f" schema: {schema_as_string}"
                                ),
                                "$ref": "#/components/schemas/execute-connector_Response",
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def get_operation(
      entity: str,
      schema_as_string: str = "",
      tool_name: str = "",
      tool_instructions: str = "",
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"Get {entity}",
            "description": (
                f"Returns the details of the {entity}. {tool_instructions}"
            ),
            "operationId": f"{tool_name}_get_{entity}",
            "x-operation": "GET_ENTITY",
            "x-entity": f"{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": f"#/components/schemas/get_{entity}_Request"
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "description": (
                                    f"Returns {entity} of json schema:"
                                    f" {schema_as_string}"
                                ),
                                "$ref": "#/components/schemas/execute-connector_Response",
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def create_operation(
      entity: str, tool_name: str = "", tool_instructions: str = ""
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"Creates a new {entity}",
            "description": f"Creates a new {entity}. {tool_instructions}",
            "x-operation": "CREATE_ENTITY",
            "x-entity": f"{entity}",
            "operationId": f"{tool_name}_create_{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": (
                                f"#/components/schemas/create_{entity}_Request"
                            )
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/execute-connector_Response"
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def update_operation(
      entity: str, tool_name: str = "", tool_instructions: str = ""
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"Updates the {entity}",
            "description": f"Updates the {entity}. {tool_instructions}",
            "x-operation": "UPDATE_ENTITY",
            "x-entity": f"{entity}",
            "operationId": f"{tool_name}_update_{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": (
                                f"#/components/schemas/update_{entity}_Request"
                            )
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/execute-connector_Response"
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def delete_operation(
      entity: str, tool_name: str = "", tool_instructions: str = ""
  ) -> Dict[str, Any]:
    return {
        "post": {
            "summary": f"Delete the {entity}",
            "description": f"Deletes the {entity}. {tool_instructions}",
            "x-operation": "DELETE_ENTITY",
            "x-entity": f"{entity}",
            "operationId": f"{tool_name}_delete_{entity}",
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "$ref": (
                                f"#/components/schemas/delete_{entity}_Request"
                            )
                        }
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Success response",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/execute-connector_Response"
                            }
                        }
                    },
                }
            },
        }
    }

  @staticmethod
  def create_operation_request(entity: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "connectorInputPayload",
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "connectorInputPayload": {
                "$ref": f"#/components/schemas/connectorInputPayload_{entity}"
            },
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
        },
    }

  @staticmethod
  def update_operation_request(entity: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "connectorInputPayload",
            "entityId",
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "connectorInputPayload": {
                "$ref": f"#/components/schemas/connectorInputPayload_{entity}"
            },
            "entityId": {"$ref": "#/components/schemas/entityId"},
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
        },
    }

  @staticmethod
  def get_operation_request() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "entityId",
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "entityId": {"$ref": "#/components/schemas/entityId"},
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
        },
    }

  @staticmethod
  def delete_operation_request() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "entityId",
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "entityId": {"$ref": "#/components/schemas/entityId"},
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
        },
    }

  @staticmethod
  def list_operation_request() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "entity",
        ],
        "properties": {
            "filterClause": {"$ref": "#/components/schemas/filterClause"},
            "pageSize": {"$ref": "#/components/schemas/pageSize"},
            "pageToken": {"$ref": "#/components/schemas/pageToken"},
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "entity": {"$ref": "#/components/schemas/entity"},
        },
    }

  @staticmethod
  def action_request(action: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "action",
            "connectorInputPayload",
        ],
        "properties": {
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "action": {"$ref": "#/components/schemas/action"},
            "connectorInputPayload": {
                "$ref": f"#/components/schemas/connectorInputPayload_{action}"
            },
        },
    }

  @staticmethod
  def action_response(action: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "connectorOutputPayload": {
                "$ref": f"#/components/schemas/connectorOutputPayload_{action}"
            },
        },
    }

  @staticmethod
  def execute_custom_query_request() -> Dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "operation",
            "connectionName",
            "serviceName",
            "host",
            "action",
            "query",
            "timeout",
            "pageSize",
        ],
        "properties": {
            "operation": {"$ref": "#/components/schemas/operation"},
            "connectionName": {"$ref": "#/components/schemas/connectionName"},
            "serviceName": {"$ref": "#/components/schemas/serviceName"},
            "host": {"$ref": "#/components/schemas/host"},
            "action": {"$ref": "#/components/schemas/action"},
            "query": {"$ref": "#/components/schemas/query"},
            "timeout": {"$ref": "#/components/schemas/timeout"},
            "pageSize": {"$ref": "#/components/schemas/pageSize"},
        },
    }

  def connector_payload(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
    return self._convert_json_schema_to_openapi_schema(json_schema)

  def _convert_json_schema_to_openapi_schema(self, json_schema):
    """Converts a JSON schema dictionary to an OpenAPI schema dictionary, handling variable types, properties, items, nullable, and description.

    Args:
        json_schema (dict): The input JSON schema dictionary.

    Returns:
        dict: The converted OpenAPI schema dictionary.
    """
    openapi_schema = {}

    if "description" in json_schema:
      openapi_schema["description"] = json_schema["description"]

    if "type" in json_schema:
      if isinstance(json_schema["type"], list):
        if "null" in json_schema["type"]:
          openapi_schema["nullable"] = True
          other_types = [t for t in json_schema["type"] if t != "null"]
          if other_types:
            openapi_schema["type"] = other_types[0]
        else:
          openapi_schema["type"] = json_schema["type"][0]
      else:
        openapi_schema["type"] = json_schema["type"]

    if openapi_schema.get("type") == "object" and "properties" in json_schema:
      openapi_schema["properties"] = {}
      for prop_name, prop_schema in json_schema["properties"].items():
        openapi_schema["properties"][prop_name] = (
            self._convert_json_schema_to_openapi_schema(prop_schema)
        )

    elif openapi_schema.get("type") == "array" and "items" in json_schema:
      if isinstance(json_schema["items"], list):
        openapi_schema["items"] = [
            self._convert_json_schema_to_openapi_schema(item)
            for item in json_schema["items"]
        ]
      else:
        openapi_schema["items"] = self._convert_json_schema_to_openapi_schema(
            json_schema["items"]
        )

    return openapi_schema

  def _get_access_token(self) -> str:
    """Gets the access token for the service account.

    Returns:
        The access token.
    """
    if self.credential_cache and not self.credential_cache.expired:
      return self.credential_cache.token

    if self.service_account_json:
      credentials = service_account.Credentials.from_service_account_info(
          json.loads(self.service_account_json),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
    else:
      try:
        credentials, _ = default_service_credential()
      except:
        credentials = None

    if not credentials:
      raise ValueError(
          "Please provide a service account that has the required permissions"
          " to access the connection."
      )

    credentials.refresh(Request())
    self.credential_cache = credentials
    return credentials.token

  def _execute_api_call(self, url):
    """Executes an API call to the given URL.

    Args:
        url (str): The URL to call.

    Returns:
        requests.Response: The response object from the API call.

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request error.
        Exception: For any other unexpected errors.
    """
    try:
      headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {self._get_access_token()}",
      }

      response = requests.get(url, headers=headers)
      response.raise_for_status()
      return response

    except google.auth.exceptions.DefaultCredentialsError as e:
      raise PermissionError(f"Credentials error: {e}") from e

    except requests.exceptions.RequestException as e:
      if (
          "404" in str(e)
          or "Not found" in str(e)
          or "400" in str(e)
          or "Bad request" in str(e)
      ):
        raise ValueError(
            "Invalid request. Please check the provided"
            f" values of project({self.project}), location({self.location}),"
            f" connection({self.connection})."
        ) from e
      raise ValueError(f"Request error: {e}") from e

    except Exception as e:
      raise Exception(f"An unexpected error occurred: {e}") from e

  def _poll_operation(self, operation_id: str) -> Dict[str, Any]:
    """Polls an operation until it is done.

    Args:
        operation_id: The ID of the operation to poll.

    Returns:
        The final response of the operation.

    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request error.
        Exception: For any other unexpected errors.
    """
    operation_done: bool = False
    operation_response: Dict[str, Any] = {}
    while not operation_done:
      get_operation_url = f"{self.connector_url}/v1/{operation_id}"
      response = self._execute_api_call(get_operation_url)
      operation_response = response.json()
      operation_done = operation_response.get("done", False)
      time.sleep(1)
    return operation_response



================================================
FILE: src/google/adk/tools/application_integration_tool/clients/integration_client.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import List, Optional
from google.adk.tools.application_integration_tool.clients.connections_client import ConnectionsClient
import google.auth
from google.auth import default as default_service_credential
import google.auth.transport.requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import requests


class IntegrationClient:
  """A client for interacting with Google Cloud Application Integration.

  This class provides methods for retrieving OpenAPI spec for an integration or
  a connection.
  """

  def __init__(
      self,
      project: str,
      location: str,
      integration: Optional[str] = None,
      triggers: List[str] = None,
      connection: Optional[str] = None,
      entity_operations: Optional[dict[str, list[str]]] = None,
      actions: Optional[list[str]] = None,
      service_account_json: Optional[str] = None,
  ):
    """Initializes the ApplicationIntegrationClient.

    Args:
        project: The Google Cloud project ID.
        location: The Google Cloud location (e.g., us-central1).
        integration: The integration name.
        triggers: The list of trigger IDs for the integration.
        connection: The connection name.
        entity_operations: A dictionary mapping entity names to a list of
          operations (e.g., LIST, CREATE, UPDATE, DELETE, GET).
        actions: List of actions.
        service_account_json: The service account configuration as a dictionary.
          Required if not using default service credential. Used for fetching
          connection details.
    """
    self.project = project
    self.location = location
    self.integration = integration
    self.triggers = triggers
    self.connection = connection
    self.entity_operations = (
        entity_operations if entity_operations is not None else {}
    )
    self.actions = actions if actions is not None else []
    self.service_account_json = service_account_json
    self.credential_cache = None

  def get_openapi_spec_for_integration(self):
    """Gets the OpenAPI spec for the integration.

    Returns:
        dict: The OpenAPI spec as a dictionary.
    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request error or processing error.
        Exception: For any other unexpected errors.
    """
    try:
      url = f"https://{self.location}-integrations.googleapis.com/v1/projects/{self.project}/locations/{self.location}:generateOpenApiSpec"
      headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {self._get_access_token()}",
      }
      data = {
          "apiTriggerResources": [
              {
                  "integrationResource": self.integration,
                  "triggerId": self.triggers,
              },
          ],
          "fileFormat": "JSON",
      }
      response = requests.post(url, headers=headers, json=data)
      response.raise_for_status()
      spec = response.json().get("openApiSpec", {})
      return json.loads(spec)
    except google.auth.exceptions.DefaultCredentialsError as e:
      raise PermissionError(f"Credentials error: {e}") from e
    except requests.exceptions.RequestException as e:
      if (
          "404" in str(e)
          or "Not found" in str(e)
          or "400" in str(e)
          or "Bad request" in str(e)
      ):
        raise ValueError(
            "Invalid request. Please check the provided values of"
            f" project({self.project}), location({self.location}),"
            f" integration({self.integration}) and trigger({self.triggers})."
        ) from e
      raise ValueError(f"Request error: {e}") from e
    except Exception as e:
      raise Exception(f"An unexpected error occurred: {e}") from e

  def get_openapi_spec_for_connection(self, tool_name="", tool_instructions=""):
    """Gets the OpenAPI spec for the connection.

    Returns:
        dict: The OpenAPI spec as a dictionary.
    Raises:
        ValueError: If there's an error retrieving the OpenAPI spec.
        PermissionError: If there are credential issues.
        Exception: For any other unexpected errors.
    """
    # Application Integration needs to be provisioned in the same region as connection and an integration with name "ExecuteConnection" and trigger "api_trigger/ExecuteConnection" should be created as per the documentation.
    integration_name = "ExecuteConnection"
    connections_client = ConnectionsClient(
        self.project,
        self.location,
        self.connection,
        self.service_account_json,
    )
    if not self.entity_operations and not self.actions:
      raise ValueError(
          "No entity operations or actions provided. Please provide at least"
          " one of them."
      )
    connector_spec = connections_client.get_connector_base_spec()
    for entity, operations in self.entity_operations.items():
      schema, supported_operations = (
          connections_client.get_entity_schema_and_operations(entity)
      )
      if not operations:
        operations = supported_operations
      json_schema_as_string = json.dumps(schema)
      entity_lower = entity
      connector_spec["components"]["schemas"][
          f"connectorInputPayload_{entity_lower}"
      ] = connections_client.connector_payload(schema)
      for operation in operations:
        operation_lower = operation.lower()
        path = f"/v2/projects/{self.project}/locations/{self.location}/integrations/{integration_name}:execute?triggerId=api_trigger/{integration_name}#{operation_lower}_{entity_lower}"
        if operation_lower == "create":
          connector_spec["paths"][path] = connections_client.create_operation(
              entity_lower, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"create_{entity_lower}_Request"
          ] = connections_client.create_operation_request(entity_lower)
        elif operation_lower == "update":
          connector_spec["paths"][path] = connections_client.update_operation(
              entity_lower, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"update_{entity_lower}_Request"
          ] = connections_client.update_operation_request(entity_lower)
        elif operation_lower == "delete":
          connector_spec["paths"][path] = connections_client.delete_operation(
              entity_lower, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"delete_{entity_lower}_Request"
          ] = connections_client.delete_operation_request()
        elif operation_lower == "list":
          connector_spec["paths"][path] = connections_client.list_operation(
              entity_lower, json_schema_as_string, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"list_{entity_lower}_Request"
          ] = connections_client.list_operation_request()
        elif operation_lower == "get":
          connector_spec["paths"][path] = connections_client.get_operation(
              entity_lower, json_schema_as_string, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"get_{entity_lower}_Request"
          ] = connections_client.get_operation_request()
        else:
          raise ValueError(
              f"Invalid operation: {operation} for entity: {entity}"
          )
    for action in self.actions:
      action_details = connections_client.get_action_schema(action)
      input_schema = action_details["inputSchema"]
      output_schema = action_details["outputSchema"]
      # Remove spaces from the display name to generate valid spec
      action_display_name = action_details["displayName"].replace(" ", "")
      operation = "EXECUTE_ACTION"
      if action == "ExecuteCustomQuery":
        connector_spec["components"]["schemas"][
            f"{action_display_name}_Request"
        ] = connections_client.execute_custom_query_request()
        operation = "EXECUTE_QUERY"
      else:
        connector_spec["components"]["schemas"][
            f"{action_display_name}_Request"
        ] = connections_client.action_request(action_display_name)
        connector_spec["components"]["schemas"][
            f"connectorInputPayload_{action_display_name}"
        ] = connections_client.connector_payload(input_schema)
      connector_spec["components"]["schemas"][
          f"connectorOutputPayload_{action_display_name}"
      ] = connections_client.connector_payload(output_schema)
      connector_spec["components"]["schemas"][
          f"{action_display_name}_Response"
      ] = connections_client.action_response(action_display_name)
      path = f"/v2/projects/{self.project}/locations/{self.location}/integrations/{integration_name}:execute?triggerId=api_trigger/{integration_name}#{action}"
      connector_spec["paths"][path] = connections_client.get_action_operation(
          action, operation, action_display_name, tool_name, tool_instructions
      )
    return connector_spec

  def _get_access_token(self) -> str:
    """Gets the access token for the service account or using default credentials.

    Returns:
        The access token.
    """
    if self.credential_cache and not self.credential_cache.expired:
      return self.credential_cache.token

    if self.service_account_json:
      credentials = service_account.Credentials.from_service_account_info(
          json.loads(self.service_account_json),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
    else:
      try:
        credentials, _ = default_service_credential()
      except:
        credentials = None

    if not credentials:
      raise ValueError(
          "Please provide a service account that has the required permissions"
          " to access the connection."
      )

    credentials.refresh(Request())
    self.credential_cache = credentials
    return credentials.token



================================================
FILE: src/google/adk/tools/google_api_tool/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
__all__ = [
    'bigquery_tool_set',
    'calendar_tool_set',
    'gmail_tool_set',
    'youtube_tool_set',
    'slides_tool_set',
    'sheets_tool_set',
    'docs_tool_set',
]

# Nothing is imported here automatically
# Each tool set will only be imported when accessed

_bigquery_tool_set = None
_calendar_tool_set = None
_gmail_tool_set = None
_youtube_tool_set = None
_slides_tool_set = None
_sheets_tool_set = None
_docs_tool_set = None


def __getattr__(name):
  global _bigquery_tool_set, _calendar_tool_set, _gmail_tool_set, _youtube_tool_set, _slides_tool_set, _sheets_tool_set, _docs_tool_set

  match name:
    case 'bigquery_tool_set':
      if _bigquery_tool_set is None:
        from .google_api_tool_sets import bigquery_tool_set as bigquery

        _bigquery_tool_set = bigquery
      return _bigquery_tool_set

    case 'calendar_tool_set':
      if _calendar_tool_set is None:
        from .google_api_tool_sets import calendar_tool_set as calendar

        _calendar_tool_set = calendar
      return _calendar_tool_set

    case 'gmail_tool_set':
      if _gmail_tool_set is None:
        from .google_api_tool_sets import gmail_tool_set as gmail

        _gmail_tool_set = gmail
      return _gmail_tool_set

    case 'youtube_tool_set':
      if _youtube_tool_set is None:
        from .google_api_tool_sets import youtube_tool_set as youtube

        _youtube_tool_set = youtube
      return _youtube_tool_set

    case 'slides_tool_set':
      if _slides_tool_set is None:
        from .google_api_tool_sets import slides_tool_set as slides

        _slides_tool_set = slides
      return _slides_tool_set

    case 'sheets_tool_set':
      if _sheets_tool_set is None:
        from .google_api_tool_sets import sheets_tool_set as sheets

        _sheets_tool_set = sheets
      return _sheets_tool_set

    case 'docs_tool_set':
      if _docs_tool_set is None:
        from .google_api_tool_sets import docs_tool_set as docs

        _docs_tool_set = docs
      return _docs_tool_set



================================================
FILE: src/google/adk/tools/google_api_tool/google_api_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from typing import Dict
from typing import Optional

from google.genai.types import FunctionDeclaration
from typing_extensions import override

from ...auth import AuthCredential
from ...auth import AuthCredentialTypes
from ...auth import OAuth2Auth
from .. import BaseTool
from ..openapi_tool import RestApiTool
from ..tool_context import ToolContext


class GoogleApiTool(BaseTool):

  def __init__(self, rest_api_tool: RestApiTool):
    super().__init__(
        name=rest_api_tool.name,
        description=rest_api_tool.description,
        is_long_running=rest_api_tool.is_long_running,
    )
    self.rest_api_tool = rest_api_tool

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    return self.rest_api_tool._get_declaration()

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: Optional[ToolContext]
  ) -> Dict[str, Any]:
    return await self.rest_api_tool.run_async(
        args=args, tool_context=tool_context
    )

  def configure_auth(self, client_id: str, client_secret: str):
    self.rest_api_tool.auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id=client_id,
            client_secret=client_secret,
        ),
    )



================================================
FILE: src/google/adk/tools/google_api_tool/google_api_tool_set.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
import os
from typing import Any
from typing import Final
from typing import List
from typing import Optional
from typing import Type

from ...auth import OpenIdConnectWithConfig
from ..openapi_tool import OpenAPIToolset
from ..openapi_tool import RestApiTool
from .google_api_tool import GoogleApiTool
from .googleapi_to_openapi_converter import GoogleApiToOpenApiConverter


class GoogleApiToolSet:
  """Google API Tool Set."""

  def __init__(self, tools: List[RestApiTool]):
    self.tools: Final[List[GoogleApiTool]] = [
        GoogleApiTool(tool) for tool in tools
    ]

  def get_tools(self) -> List[GoogleApiTool]:
    """Get all tools in the toolset."""
    return self.tools

  def get_tool(self, tool_name: str) -> Optional[GoogleApiTool]:
    """Get a tool by name."""
    matching_tool = filter(lambda t: t.name == tool_name, self.tools)
    return next(matching_tool, None)

  @staticmethod
  def _load_tool_set_with_oidc_auth(
      spec_file: Optional[str] = None,
      spec_dict: Optional[dict[str, Any]] = None,
      scopes: Optional[list[str]] = None,
  ) -> OpenAPIToolset:
    spec_str = None
    if spec_file:
      # Get the frame of the caller
      caller_frame = inspect.stack()[1]
      # Get the filename of the caller
      caller_filename = caller_frame.filename
      # Get the directory of the caller
      caller_dir = os.path.dirname(os.path.abspath(caller_filename))
      # Join the directory path with the filename
      yaml_path = os.path.join(caller_dir, spec_file)
      with open(yaml_path, 'r', encoding='utf-8') as file:
        spec_str = file.read()
    tool_set = OpenAPIToolset(
        spec_dict=spec_dict,
        spec_str=spec_str,
        spec_str_type='yaml',
        auth_scheme=OpenIdConnectWithConfig(
            authorization_endpoint=(
                'https://accounts.google.com/o/oauth2/v2/auth'
            ),
            token_endpoint='https://oauth2.googleapis.com/token',
            userinfo_endpoint=(
                'https://openidconnect.googleapis.com/v1/userinfo'
            ),
            revocation_endpoint='https://oauth2.googleapis.com/revoke',
            token_endpoint_auth_methods_supported=[
                'client_secret_post',
                'client_secret_basic',
            ],
            grant_types_supported=['authorization_code'],
            scopes=scopes,
        ),
    )
    return tool_set

  def configure_auth(self, client_id: str, client_secret: str):
    for tool in self.tools:
      tool.configure_auth(client_id, client_secret)

  @classmethod
  def load_tool_set(
      cls: Type[GoogleApiToolSet],
      api_name: str,
      api_version: str,
  ) -> GoogleApiToolSet:
    spec_dict = GoogleApiToOpenApiConverter(api_name, api_version).convert()
    scope = list(
        spec_dict['components']['securitySchemes']['oauth2']['flows'][
            'authorizationCode'
        ]['scopes'].keys()
    )[0]
    return cls(
        cls._load_tool_set_with_oidc_auth(
            spec_dict=spec_dict, scopes=[scope]
        ).get_tools()
    )



================================================
FILE: src/google/adk/tools/google_api_tool/google_api_tool_sets.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

from .google_api_tool_set import GoogleApiToolSet

logger = logging.getLogger(__name__)

_bigquery_tool_set = None
_calendar_tool_set = None
_gmail_tool_set = None
_youtube_tool_set = None
_slides_tool_set = None
_sheets_tool_set = None
_docs_tool_set = None


def __getattr__(name):
  """This method dynamically loads and returns GoogleApiToolSet instances for

  various Google APIs. It uses a lazy loading approach, initializing each
  tool set only when it is first requested. This avoids unnecessary loading
  of tool sets that are not used in a given session.

  Args:
      name (str): The name of the tool set to retrieve (e.g.,
        "bigquery_tool_set").

  Returns:
      GoogleApiToolSet: The requested tool set instance.

  Raises:
      AttributeError: If the requested tool set name is not recognized.
  """
  global _bigquery_tool_set, _calendar_tool_set, _gmail_tool_set, _youtube_tool_set, _slides_tool_set, _sheets_tool_set, _docs_tool_set

  match name:
    case "bigquery_tool_set":
      if _bigquery_tool_set is None:
        _bigquery_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="bigquery",
            api_version="v2",
        )

      return _bigquery_tool_set

    case "calendar_tool_set":
      if _calendar_tool_set is None:
        _calendar_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="calendar",
            api_version="v3",
        )

      return _calendar_tool_set

    case "gmail_tool_set":
      if _gmail_tool_set is None:
        _gmail_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="gmail",
            api_version="v1",
        )

      return _gmail_tool_set

    case "youtube_tool_set":
      if _youtube_tool_set is None:
        _youtube_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="youtube",
            api_version="v3",
        )

      return _youtube_tool_set

    case "slides_tool_set":
      if _slides_tool_set is None:
        _slides_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="slides",
            api_version="v1",
        )

      return _slides_tool_set

    case "sheets_tool_set":
      if _sheets_tool_set is None:
        _sheets_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="sheets",
            api_version="v4",
        )

      return _sheets_tool_set

    case "docs_tool_set":
      if _docs_tool_set is None:
        _docs_tool_set = GoogleApiToolSet.load_tool_set(
            api_name="docs",
            api_version="v1",
        )

      return _docs_tool_set



================================================
FILE: src/google/adk/tools/google_api_tool/googleapi_to_openapi_converter.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# Google API client
from googleapiclient.discovery import build
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GoogleApiToOpenApiConverter:
  """Converts Google API Discovery documents to OpenAPI v3 format."""

  def __init__(self, api_name: str, api_version: str):
    """Initialize the converter with the API name and version.

    Args:
        api_name: The name of the Google API (e.g., "calendar")
        api_version: The version of the API (e.g., "v3")
    """
    self.api_name = api_name
    self.api_version = api_version
    self.google_api_resource = None
    self.google_api_spec = None
    self.openapi_spec = {
        "openapi": "3.0.0",
        "info": {},
        "servers": [],
        "paths": {},
        "components": {"schemas": {}, "securitySchemes": {}},
    }

  def fetch_google_api_spec(self) -> None:
    """Fetches the Google API specification using discovery service."""
    try:
      logger.info(
          "Fetching Google API spec for %s %s", self.api_name, self.api_version
      )
      # Build a resource object for the specified API
      self.google_api_resource = build(self.api_name, self.api_version)

      # Access the underlying API discovery document
      self.google_api_spec = self.google_api_resource._rootDesc

      if not self.google_api_spec:
        raise ValueError("Failed to retrieve API specification")

      logger.info("Successfully fetched %s API specification", self.api_name)
    except HttpError as e:
      logger.error("HTTP Error: %s", e)
      raise
    except Exception as e:
      logger.error("Error fetching API spec: %s", e)
      raise

  def convert(self) -> Dict[str, Any]:
    """Convert the Google API spec to OpenAPI v3 format.

    Returns:
        Dict containing the converted OpenAPI v3 specification
    """
    if not self.google_api_spec:
      self.fetch_google_api_spec()

    # Convert basic API information
    self._convert_info()

    # Convert server information
    self._convert_servers()

    # Convert authentication/authorization schemes
    self._convert_security_schemes()

    # Convert schemas (models)
    self._convert_schemas()

    # Convert endpoints/paths
    self._convert_resources(self.google_api_spec.get("resources", {}))

    # Convert top-level methods, if any
    self._convert_methods(self.google_api_spec.get("methods", {}), "/")

    return self.openapi_spec

  def _convert_info(self) -> None:
    """Convert basic API information."""
    self.openapi_spec["info"] = {
        "title": self.google_api_spec.get("title", f"{self.api_name} API"),
        "description": self.google_api_spec.get("description", ""),
        "version": self.google_api_spec.get("version", self.api_version),
        "contact": {},
        "termsOfService": self.google_api_spec.get("documentationLink", ""),
    }

    # Add documentation links if available
    docs_link = self.google_api_spec.get("documentationLink")
    if docs_link:
      self.openapi_spec["externalDocs"] = {
          "description": "API Documentation",
          "url": docs_link,
      }

  def _convert_servers(self) -> None:
    """Convert server information."""
    base_url = self.google_api_spec.get(
        "rootUrl", ""
    ) + self.google_api_spec.get("servicePath", "")

    # Remove trailing slash if present
    if base_url.endswith("/"):
      base_url = base_url[:-1]

    self.openapi_spec["servers"] = [{
        "url": base_url,
        "description": f"{self.api_name} {self.api_version} API",
    }]

  def _convert_security_schemes(self) -> None:
    """Convert authentication and authorization schemes."""
    auth = self.google_api_spec.get("auth", {})
    oauth2 = auth.get("oauth2", {})

    if oauth2:
      # Handle OAuth2
      scopes = oauth2.get("scopes", {})
      formatted_scopes = {}

      for scope, scope_info in scopes.items():
        formatted_scopes[scope] = scope_info.get("description", "")

      self.openapi_spec["components"]["securitySchemes"]["oauth2"] = {
          "type": "oauth2",
          "description": "OAuth 2.0 authentication",
          "flows": {
              "authorizationCode": {
                  "authorizationUrl": (
                      "https://accounts.google.com/o/oauth2/auth"
                  ),
                  "tokenUrl": "https://oauth2.googleapis.com/token",
                  "scopes": formatted_scopes,
              }
          },
      }

    # Add API key authentication (most Google APIs support this)
    self.openapi_spec["components"]["securitySchemes"]["apiKey"] = {
        "type": "apiKey",
        "in": "query",
        "name": "key",
        "description": "API key for accessing this API",
    }

    # Create global security requirement
    self.openapi_spec["security"] = [
        {"oauth2": list(formatted_scopes.keys())} if oauth2 else {},
        {"apiKey": []},
    ]

  def _convert_schemas(self) -> None:
    """Convert schema definitions (models)."""
    schemas = self.google_api_spec.get("schemas", {})

    for schema_name, schema_def in schemas.items():
      converted_schema = self._convert_schema_object(schema_def)
      self.openapi_spec["components"]["schemas"][schema_name] = converted_schema

  def _convert_schema_object(
      self, schema_def: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Recursively convert a Google API schema object to OpenAPI schema.

    Args:
        schema_def: Google API schema definition

    Returns:
        Converted OpenAPI schema object
    """
    result = {}

    # Convert the type
    if "type" in schema_def:
      gtype = schema_def["type"]
      if gtype == "object":
        result["type"] = "object"

        # Handle properties
        if "properties" in schema_def:
          result["properties"] = {}
          for prop_name, prop_def in schema_def["properties"].items():
            result["properties"][prop_name] = self._convert_schema_object(
                prop_def
            )

        # Handle required fields
        required_fields = []
        for prop_name, prop_def in schema_def.get("properties", {}).items():
          if prop_def.get("required", False):
            required_fields.append(prop_name)
        if required_fields:
          result["required"] = required_fields

      elif gtype == "array":
        result["type"] = "array"
        if "items" in schema_def:
          result["items"] = self._convert_schema_object(schema_def["items"])

      elif gtype == "any":
        # OpenAPI doesn't have direct "any" type
        # Use oneOf with multiple options as alternative
        result["oneOf"] = [
            {"type": "object"},
            {"type": "array"},
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
        ]

      else:
        # Handle other primitive types
        result["type"] = gtype

    # Handle references
    if "$ref" in schema_def:
      ref = schema_def["$ref"]
      # Google refs use "#" at start, OpenAPI uses "#/components/schemas/"
      if ref.startswith("#"):
        ref = ref.replace("#", "#/components/schemas/")
      else:
        ref = "#/components/schemas/" + ref
      result["$ref"] = ref

    # Handle format
    if "format" in schema_def:
      result["format"] = schema_def["format"]

    # Handle enum values
    if "enum" in schema_def:
      result["enum"] = schema_def["enum"]

    # Handle description
    if "description" in schema_def:
      result["description"] = schema_def["description"]

    # Handle pattern
    if "pattern" in schema_def:
      result["pattern"] = schema_def["pattern"]

    # Handle default value
    if "default" in schema_def:
      result["default"] = schema_def["default"]

    return result

  def _convert_resources(
      self, resources: Dict[str, Any], parent_path: str = ""
  ) -> None:
    """Recursively convert all resources and their methods.

    Args:
        resources: Dictionary of resources from the Google API spec
        parent_path: The parent path prefix for nested resources
    """
    for resource_name, resource_data in resources.items():
      # Process methods for this resource
      resource_path = f"{parent_path}/{resource_name}"
      methods = resource_data.get("methods", {})
      self._convert_methods(methods, resource_path)

      # Process nested resources recursively
      nested_resources = resource_data.get("resources", {})
      if nested_resources:
        self._convert_resources(nested_resources, resource_path)

  def _convert_methods(
      self, methods: Dict[str, Any], resource_path: str
  ) -> None:
    """Convert methods for a specific resource path.

    Args:
        methods: Dictionary of methods from the Google API spec
        resource_path: The path of the resource these methods belong to
    """
    for method_name, method_data in methods.items():
      http_method = method_data.get("httpMethod", "GET").lower()

      # Determine the actual endpoint path
      # Google often has the format something like 'users.messages.list'
      # flatPath is preferred as it provides the actual path, while path
      # might contain variables like {+projectId}
      rest_path = method_data.get("flatPath", method_data.get("path", "/"))
      if not rest_path.startswith("/"):
        rest_path = "/" + rest_path

      path_params = self._extract_path_parameters(rest_path)

      # Create path entry if it doesn't exist
      if rest_path not in self.openapi_spec["paths"]:
        self.openapi_spec["paths"][rest_path] = {}

      # Add the operation for this method
      self.openapi_spec["paths"][rest_path][http_method] = (
          self._convert_operation(method_data, path_params)
      )

  def _extract_path_parameters(self, path: str) -> List[str]:
    """Extract path parameters from a URL path.

    Args:
        path: The URL path with path parameters

    Returns:
        List of parameter names
    """
    params = []
    segments = path.split("/")

    for segment in segments:
      # Google APIs often use {param} format for path parameters
      if segment.startswith("{") and segment.endswith("}"):
        param_name = segment[1:-1]
        params.append(param_name)

    return params

  def _convert_operation(
      self, method_data: Dict[str, Any], path_params: List[str]
  ) -> Dict[str, Any]:
    """Convert a Google API method to an OpenAPI operation.

    Args:
        method_data: Google API method data
        path_params: List of path parameter names

    Returns:
        OpenAPI operation object
    """
    operation = {
        "operationId": method_data.get("id", ""),
        "summary": method_data.get("description", ""),
        "description": method_data.get("description", ""),
        "parameters": [],
        "responses": {
            "200": {"description": "Successful operation"},
            "400": {"description": "Bad request"},
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden"},
            "404": {"description": "Not found"},
            "500": {"description": "Server error"},
        },
    }

    # Add path parameters
    for param_name in path_params:
      param = {
          "name": param_name,
          "in": "path",
          "required": True,
          "schema": {"type": "string"},
      }
      operation["parameters"].append(param)

    # Add query parameters
    for param_name, param_data in method_data.get("parameters", {}).items():
      # Skip parameters already included in path
      if param_name in path_params:
        continue

      param = {
          "name": param_name,
          "in": "query",
          "description": param_data.get("description", ""),
          "required": param_data.get("required", False),
          "schema": self._convert_parameter_schema(param_data),
      }
      operation["parameters"].append(param)

    # Handle request body
    if "request" in method_data:
      request_ref = method_data.get("request", {}).get("$ref", "")
      if request_ref:
        if request_ref.startswith("#"):
          # Convert Google's reference format to OpenAPI format
          openapi_ref = request_ref.replace("#", "#/components/schemas/")
        else:
          openapi_ref = "#/components/schemas/" + request_ref
        operation["requestBody"] = {
            "description": "Request body",
            "content": {"application/json": {"schema": {"$ref": openapi_ref}}},
            "required": True,
        }

    # Handle response body
    if "response" in method_data:
      response_ref = method_data.get("response", {}).get("$ref", "")
      if response_ref:
        if response_ref.startswith("#"):
          # Convert Google's reference format to OpenAPI format
          openapi_ref = response_ref.replace("#", "#/components/schemas/")
        else:
          openapi_ref = "#/components/schemas/" + response_ref
        operation["responses"]["200"]["content"] = {
            "application/json": {"schema": {"$ref": openapi_ref}}
        }

    # Add scopes if available
    scopes = method_data.get("scopes", [])
    if scopes:
      # Add method-specific security requirement if different from global
      operation["security"] = [{"oauth2": scopes}]

    return operation

  def _convert_parameter_schema(
      self, param_data: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Convert a parameter definition to an OpenAPI schema.

    Args:
        param_data: Google API parameter data

    Returns:
        OpenAPI schema for the parameter
    """
    schema = {}

    # Convert type
    param_type = param_data.get("type", "string")
    schema["type"] = param_type

    # Handle enum values
    if "enum" in param_data:
      schema["enum"] = param_data["enum"]

    # Handle format
    if "format" in param_data:
      schema["format"] = param_data["format"]

    # Handle default value
    if "default" in param_data:
      schema["default"] = param_data["default"]

    # Handle pattern
    if "pattern" in param_data:
      schema["pattern"] = param_data["pattern"]

    return schema

  def save_openapi_spec(self, output_path: str) -> None:
    """Save the OpenAPI specification to a file.

    Args:
        output_path: Path where the OpenAPI spec should be saved
    """
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(self.openapi_spec, f, indent=2)
    logger.info("OpenAPI specification saved to %s", output_path)


def main():
  """Command line interface for the converter."""
  parser = argparse.ArgumentParser(
      description=(
          "Convert Google API Discovery documents to OpenAPI v3 specifications"
      )
  )
  parser.add_argument(
      "api_name", help="Name of the Google API (e.g., 'calendar')"
  )
  parser.add_argument("api_version", help="Version of the API (e.g., 'v3')")
  parser.add_argument(
      "--output",
      "-o",
      default="openapi_spec.json",
      help="Output file path for the OpenAPI specification",
  )

  args = parser.parse_args()

  try:
    # Create and run the converter
    converter = GoogleApiToOpenApiConverter(args.api_name, args.api_version)
    converter.convert()
    converter.save_openapi_spec(args.output)
    print(
        f"Successfully converted {args.api_name} {args.api_version} to"
        " OpenAPI v3"
    )
    print(f"Output saved to {args.output}")
  except Exception as e:
    logger.error("Conversion failed: %s", e)
    return 1

  return 0


if __name__ == "__main__":
  main()



================================================
FILE: src/google/adk/tools/mcp_tool/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = []

try:
  from .conversion_utils import adk_to_mcp_tool_type, gemini_to_json_schema
  from .mcp_tool import MCPTool
  from .mcp_toolset import MCPToolset

  __all__.extend([
      'adk_to_mcp_tool_type',
      'gemini_to_json_schema',
      'MCPTool',
      'MCPToolset',
  ])

except ImportError as e:
  import logging
  import sys

  logger = logging.getLogger(__name__)

  if sys.version_info < (3, 10):
    logger.warning(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    )
  else:
    logger.debug('MCP Tool is not installed')
    logger.debug(e)



================================================
FILE: src/google/adk/tools/mcp_tool/conversion_utils.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict
from google.genai.types import Schema, Type
import mcp.types as mcp_types
from ..base_tool import BaseTool


def adk_to_mcp_tool_type(tool: BaseTool) -> mcp_types.Tool:
  """Convert a Tool in ADK into MCP tool type.

  This function transforms an ADK tool definition into its equivalent
  representation in the MCP (Model Context Protocol) system.

  Args:
      tool: The ADK tool to convert. It should be an instance of a class derived
        from `BaseTool`.

  Returns:
      An object of MCP Tool type, representing the converted tool.

  Examples:
      # Assuming 'my_tool' is an instance of a BaseTool derived class
      mcp_tool = adk_to_mcp_tool_type(my_tool)
      print(mcp_tool)
  """
  tool_declaration = tool._get_declaration()
  if not tool_declaration:
    input_schema = {}
  else:
    input_schema = gemini_to_json_schema(tool._get_declaration().parameters)
  return mcp_types.Tool(
      name=tool.name,
      description=tool.description,
      inputSchema=input_schema,
  )


def gemini_to_json_schema(gemini_schema: Schema) -> Dict[str, Any]:
  """Converts a Gemini Schema object into a JSON Schema dictionary.

  Args:
      gemini_schema: An instance of the Gemini Schema class.

  Returns:
      A dictionary representing the equivalent JSON Schema.

  Raises:
      TypeError: If the input is not an instance of the expected Schema class.
      ValueError: If an invalid Gemini Type enum value is encountered.
  """
  if not isinstance(gemini_schema, Schema):
    raise TypeError(
        f"Input must be an instance of Schema, got {type(gemini_schema)}"
    )

  json_schema_dict: Dict[str, Any] = {}

  # Map Type
  gemini_type = getattr(gemini_schema, "type", None)
  if gemini_type and gemini_type != Type.TYPE_UNSPECIFIED:
    json_schema_dict["type"] = gemini_type.lower()
  else:
    json_schema_dict["type"] = "null"

  # Map Nullable
  if getattr(gemini_schema, "nullable", None) == True:
    json_schema_dict["nullable"] = True

  # --- Map direct fields ---
  direct_mappings = {
      "title": "title",
      "description": "description",
      "default": "default",
      "enum": "enum",
      "format": "format",
      "example": "example",
  }
  for gemini_key, json_key in direct_mappings.items():
    value = getattr(gemini_schema, gemini_key, None)
    if value is not None:
      json_schema_dict[json_key] = value

  # String validation
  if gemini_type == Type.STRING:
    str_mappings = {
        "pattern": "pattern",
        "min_length": "minLength",
        "max_length": "maxLength",
    }
    for gemini_key, json_key in str_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Number/Integer validation
  if gemini_type in (Type.NUMBER, Type.INTEGER):
    num_mappings = {
        "minimum": "minimum",
        "maximum": "maximum",
    }
    for gemini_key, json_key in num_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Array validation (Recursive call for items)
  if gemini_type == Type.ARRAY:
    items_schema = getattr(gemini_schema, "items", None)
    if items_schema is not None:
      json_schema_dict["items"] = gemini_to_json_schema(items_schema)

    arr_mappings = {
        "min_items": "minItems",
        "max_items": "maxItems",
    }
    for gemini_key, json_key in arr_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Object validation (Recursive call for properties)
  if gemini_type == Type.OBJECT:
    properties_dict = getattr(gemini_schema, "properties", None)
    if properties_dict is not None:
      json_schema_dict["properties"] = {
          prop_name: gemini_to_json_schema(prop_schema)
          for prop_name, prop_schema in properties_dict.items()
      }

    obj_mappings = {
        "required": "required",
        "min_properties": "minProperties",
        "max_properties": "maxProperties",
        # Note: Ignoring 'property_ordering' as it's not standard JSON Schema
    }
    for gemini_key, json_key in obj_mappings.items():
      value = getattr(gemini_schema, gemini_key, None)
      if value is not None:
        json_schema_dict[json_key] = value

  # Map anyOf (Recursive call for subschemas)
  any_of_list = getattr(gemini_schema, "any_of", None)
  if any_of_list is not None:
    json_schema_dict["anyOf"] = [
        gemini_to_json_schema(sub_schema) for sub_schema in any_of_list
    ]

  return json_schema_dict



================================================
FILE: src/google/adk/tools/mcp_tool/mcp_session_manager.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import AsyncExitStack
import functools
import sys
from typing import Any, TextIO
import anyio
from pydantic import BaseModel

try:
  from mcp import ClientSession, StdioServerParameters
  from mcp.client.sse import sse_client
  from mcp.client.stdio import stdio_client
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e


class SseServerParams(BaseModel):
  """Parameters for the MCP SSE connection.

  See MCP SSE Client documentation for more details.
  https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/sse.py
  """

  url: str
  headers: dict[str, Any] | None = None
  timeout: float = 5
  sse_read_timeout: float = 60 * 5


def retry_on_closed_resource(async_reinit_func_name: str):
  """Decorator to automatically reinitialize session and retry action.

  When MCP session was closed, the decorator will automatically recreate the
  session and retry the action with the same parameters.

  Note:
  1. async_reinit_func_name is the name of the class member function that
  reinitializes the MCP session.
  2. Both the decorated function and the async_reinit_func_name must be async
  functions.

  Usage:
  class MCPTool:
    ...
    async def create_session(self):
      self.session = ...

    @retry_on_closed_resource('create_session')
    async def use_session(self):
      await self.session.call_tool()

  Args:
    async_reinit_func_name: The name of the async function to recreate session.

  Returns:
    The decorated function.
  """

  def decorator(func):
    @functools.wraps(
        func
    )  # Preserves original function metadata (name, docstring)
    async def wrapper(self, *args, **kwargs):
      try:
        return await func(self, *args, **kwargs)
      except anyio.ClosedResourceError:
        try:
          if hasattr(self, async_reinit_func_name) and callable(
              getattr(self, async_reinit_func_name)
          ):
            async_init_fn = getattr(self, async_reinit_func_name)
            await async_init_fn()
          else:
            raise ValueError(
                f'Function {async_reinit_func_name} does not exist in decorated'
                ' class. Please check the function name in'
                ' retry_on_closed_resource decorator.'
            )
        except Exception as reinit_err:
          raise RuntimeError(
              f'Error reinitializing: {reinit_err}'
          ) from reinit_err
        return await func(self, *args, **kwargs)

    return wrapper

  return decorator


class MCPSessionManager:
  """Manages MCP client sessions.

  This class provides methods for creating and initializing MCP client sessions,
  handling different connection parameters (Stdio and SSE).
  """

  def __init__(
      self,
      connection_params: StdioServerParameters | SseServerParams,
      exit_stack: AsyncExitStack,
      errlog: TextIO = sys.stderr,
  ) -> ClientSession:
    """Initializes the MCP session manager.

    Example usage:
    ```
    mcp_session_manager = MCPSessionManager(
        connection_params=connection_params,
        exit_stack=exit_stack,
    )
    session = await mcp_session_manager.create_session()
    ```

    Args:
        connection_params: Parameters for the MCP connection (Stdio or SSE).
        exit_stack: AsyncExitStack to manage the session lifecycle.
        errlog: (Optional) TextIO stream for error logging. Use only for
          initializing a local stdio MCP session.
    """
    self.connection_params = connection_params
    self.exit_stack = exit_stack
    self.errlog = errlog

  async def create_session(self) -> ClientSession:
    return await MCPSessionManager.initialize_session(
        connection_params=self.connection_params,
        exit_stack=self.exit_stack,
        errlog=self.errlog,
    )

  @classmethod
  async def initialize_session(
      cls,
      *,
      connection_params: StdioServerParameters | SseServerParams,
      exit_stack: AsyncExitStack,
      errlog: TextIO = sys.stderr,
  ) -> ClientSession:
    """Initializes an MCP client session.

    Args:
        connection_params: Parameters for the MCP connection (Stdio or SSE).
        exit_stack: AsyncExitStack to manage the session lifecycle.
        errlog: (Optional) TextIO stream for error logging. Use only for
          initializing a local stdio MCP session.

    Returns:
        ClientSession: The initialized MCP client session.
    """
    if isinstance(connection_params, StdioServerParameters):
      client = stdio_client(server=connection_params, errlog=errlog)
    elif isinstance(connection_params, SseServerParams):
      client = sse_client(
          url=connection_params.url,
          headers=connection_params.headers,
          timeout=connection_params.timeout,
          sse_read_timeout=connection_params.sse_read_timeout,
      )
    else:
      raise ValueError(
          'Unable to initialize connection. Connection should be'
          ' StdioServerParameters or SseServerParams, but got'
          f' {connection_params}'
      )

    transports = await exit_stack.enter_async_context(client)
    session = await exit_stack.enter_async_context(ClientSession(*transports))
    await session.initialize()
    return session



================================================
FILE: src/google/adk/tools/mcp_tool/mcp_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from google.genai.types import FunctionDeclaration
from typing_extensions import override

from .mcp_session_manager import MCPSessionManager, retry_on_closed_resource

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp import ClientSession
  from mcp.types import Tool as McpBaseTool
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        "MCP Tool requires Python 3.10 or above. Please upgrade your Python"
        " version."
    ) from e
  else:
    raise e


from ..base_tool import BaseTool
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ..openapi_tool.openapi_spec_parser.rest_api_tool import to_gemini_schema
from ..tool_context import ToolContext


class MCPTool(BaseTool):
  """Turns a MCP Tool into a Vertex Agent Framework Tool.

  Internally, the tool initializes from a MCP Tool, and uses the MCP Session to
  call the tool.
  """

  def __init__(
      self,
      mcp_tool: McpBaseTool,
      mcp_session: ClientSession,
      mcp_session_manager: MCPSessionManager,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] | None = None,
  ):
    """Initializes a MCPTool.

    This tool wraps a MCP Tool interface and an active MCP Session. It invokes
    the MCP Tool through executing the tool from remote MCP Session.

    Example:
        tool = MCPTool(mcp_tool=mcp_tool, mcp_session=mcp_session)

    Args:
        mcp_tool: The MCP tool to wrap.
        mcp_session: The MCP session to use to call the tool.
        auth_scheme: The authentication scheme to use.
        auth_credential: The authentication credential to use.

    Raises:
        ValueError: If mcp_tool or mcp_session is None.
    """
    if mcp_tool is None:
      raise ValueError("mcp_tool cannot be None")
    if mcp_session is None:
      raise ValueError("mcp_session cannot be None")
    self.name = mcp_tool.name
    self.description = mcp_tool.description if mcp_tool.description else ""
    self.mcp_tool = mcp_tool
    self.mcp_session = mcp_session
    self.mcp_session_manager = mcp_session_manager
    # TODO(cheliu): Support passing auth to MCP Server.
    self.auth_scheme = auth_scheme
    self.auth_credential = auth_credential

  async def _reinitialize_session(self):
    self.mcp_session = await self.mcp_session_manager.create_session()

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Gets the function declaration for the tool.

    Returns:
        FunctionDeclaration: The Gemini function declaration for the tool.
    """
    schema_dict = self.mcp_tool.inputSchema
    parameters = to_gemini_schema(schema_dict)
    function_decl = FunctionDeclaration(
        name=self.name, description=self.description, parameters=parameters
    )
    return function_decl

  @override
  @retry_on_closed_resource("_reinitialize_session")
  async def run_async(self, *, args, tool_context: ToolContext):
    """Runs the tool asynchronously.

    Args:
        args: The arguments as a dict to pass to the tool.
        tool_context: The tool context from upper level ADK agent.

    Returns:
        Any: The response from the tool.
    """
    # TODO(cheliu): Support passing tool context to MCP Server.
    try:
      response = await self.mcp_session.call_tool(self.name, arguments=args)
      return response
    except Exception as e:
      print(e)
      raise e



================================================
FILE: src/google/adk/tools/mcp_tool/mcp_toolset.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import AsyncExitStack
import sys
from types import TracebackType
from typing import List, Optional, TextIO, Tuple, Type

from .mcp_session_manager import MCPSessionManager, SseServerParams, retry_on_closed_resource

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp import ClientSession, StdioServerParameters
  from mcp.types import ListToolsResult
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

from .mcp_tool import MCPTool


class MCPToolset:
  """Connects to a MCP Server, and retrieves MCP Tools into ADK Tools.

  Usage:
  Example 1: (using from_server helper):
  ```
  async def load_tools():
    return await MCPToolset.from_server(
      connection_params=StdioServerParameters(
          command='npx',
          args=["-y", "@modelcontextprotocol/server-filesystem"],
          )
    )

  # Use the tools in an LLM agent
  tools, exit_stack = await load_tools()
  agent = LlmAgent(
      tools=tools
  )
  ...
  await exit_stack.aclose()
  ```

  Example 2: (using `async with`):

  ```
  async def load_tools():
    async with MCPToolset(
      connection_params=SseServerParams(url="http://0.0.0.0:8090/sse")
    ) as toolset:
      tools = await toolset.load_tools()

      agent = LlmAgent(
          ...
          tools=tools
      )
  ```

  Example 3: (provide AsyncExitStack):
  ```
  async def load_tools():
    async_exit_stack = AsyncExitStack()
    toolset = MCPToolset(
      connection_params=StdioServerParameters(...),
    )
    async_exit_stack.enter_async_context(toolset)
    tools = await toolset.load_tools()
    agent = LlmAgent(
        ...
        tools=tools
    )
    ...
    await async_exit_stack.aclose()

  ```

  Attributes:
    connection_params: The connection parameters to the MCP server. Can be
      either `StdioServerParameters` or `SseServerParams`.
    exit_stack: The async exit stack to manage the connection to the MCP server.
    session: The MCP session being initialized with the connection.
  """

  def __init__(
      self,
      *,
      connection_params: StdioServerParameters | SseServerParams,
      errlog: TextIO = sys.stderr,
      exit_stack=AsyncExitStack(),
  ):
    """Initializes the MCPToolset.

    Usage:
    Example 1: (using from_server helper):
    ```
    async def load_tools():
      return await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command='npx',
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            )
      )

    # Use the tools in an LLM agent
    tools, exit_stack = await load_tools()
    agent = LlmAgent(
        tools=tools
    )
    ...
    await exit_stack.aclose()
    ```

    Example 2: (using `async with`):

    ```
    async def load_tools():
      async with MCPToolset(
        connection_params=SseServerParams(url="http://0.0.0.0:8090/sse")
      ) as toolset:
        tools = await toolset.load_tools()

        agent = LlmAgent(
            ...
            tools=tools
        )
    ```

    Example 3: (provide AsyncExitStack):
    ```
    async def load_tools():
      async_exit_stack = AsyncExitStack()
      toolset = MCPToolset(
        connection_params=StdioServerParameters(...),
      )
      async_exit_stack.enter_async_context(toolset)
      tools = await toolset.load_tools()
      agent = LlmAgent(
          ...
          tools=tools
      )
      ...
      await async_exit_stack.aclose()

    ```

    Args:
      connection_params: The connection parameters to the MCP server. Can be:
        `StdioServerParameters` for using local mcp server (e.g. using `npx` or
        `python3`); or `SseServerParams` for a local/remote SSE server.
    """
    if not connection_params:
      raise ValueError('Missing connection params in MCPToolset.')
    self.connection_params = connection_params
    self.errlog = errlog
    self.exit_stack = exit_stack

    self.session_manager = MCPSessionManager(
        connection_params=self.connection_params,
        exit_stack=self.exit_stack,
        errlog=self.errlog,
    )

  @classmethod
  async def from_server(
      cls,
      *,
      connection_params: StdioServerParameters | SseServerParams,
      async_exit_stack: Optional[AsyncExitStack] = None,
      errlog: TextIO = sys.stderr,
  ) -> Tuple[List[MCPTool], AsyncExitStack]:
    """Retrieve all tools from the MCP connection.

    Usage:
    ```
    async def load_tools():
      tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command='npx',
            args=["-y", "@modelcontextprotocol/server-filesystem"],
        )
      )
    ```

    Args:
      connection_params: The connection parameters to the MCP server.
      async_exit_stack: The async exit stack to use. If not provided, a new
        AsyncExitStack will be created.

    Returns:
      A tuple of the list of MCPTools and the AsyncExitStack.
      - tools: The list of MCPTools.
      - async_exit_stack: The AsyncExitStack used to manage the connection to
        the MCP server. Use `await async_exit_stack.aclose()` to close the
        connection when server shuts down.
    """
    async_exit_stack = async_exit_stack or AsyncExitStack()
    toolset = cls(
        connection_params=connection_params,
        exit_stack=async_exit_stack,
        errlog=errlog,
    )

    await async_exit_stack.enter_async_context(toolset)
    tools = await toolset.load_tools()
    return (tools, async_exit_stack)

  async def _initialize(self) -> ClientSession:
    """Connects to the MCP Server and initializes the ClientSession."""
    self.session = await self.session_manager.create_session()
    return self.session

  async def _exit(self):
    """Closes the connection to MCP Server."""
    await self.exit_stack.aclose()

  @retry_on_closed_resource('_initialize')
  async def load_tools(self) -> List[MCPTool]:
    """Loads all tools from the MCP Server.

    Returns:
      A list of MCPTools imported from the MCP Server.
    """
    tools_response: ListToolsResult = await self.session.list_tools()
    return [
        MCPTool(
            mcp_tool=tool,
            mcp_session=self.session,
            mcp_session_manager=self.session_manager,
        )
        for tool in tools_response.tools
    ]

  async def __aenter__(self):
    try:
      await self._initialize()
      return self
    except Exception as e:
      raise e

  async def __aexit__(
      self,
      exc_type: Optional[Type[BaseException]],
      exc: Optional[BaseException],
      tb: Optional[TracebackType],
  ) -> None:
    await self._exit()



================================================
FILE: src/google/adk/tools/openapi_tool/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .openapi_spec_parser import OpenAPIToolset
from .openapi_spec_parser import RestApiTool

__all__ = [
    'OpenAPIToolset',
    'RestApiTool',
]



================================================
FILE: src/google/adk/tools/openapi_tool/auth/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import auth_helpers

__all__ = [
    'auth_helpers',
]



================================================
FILE: src/google/adk/tools/openapi_tool/auth/auth_helpers.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

from fastapi.openapi.models import APIKey
from fastapi.openapi.models import APIKeyIn
from fastapi.openapi.models import HTTPBase
from fastapi.openapi.models import HTTPBearer
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OpenIdConnect
from fastapi.openapi.models import Schema
from pydantic import BaseModel
from pydantic import ValidationError
import requests

from ....auth.auth_credential import AuthCredential
from ....auth.auth_credential import AuthCredentialTypes
from ....auth.auth_credential import HttpAuth
from ....auth.auth_credential import HttpCredentials
from ....auth.auth_credential import OAuth2Auth
from ....auth.auth_credential import ServiceAccount
from ....auth.auth_credential import ServiceAccountCredential
from ....auth.auth_schemes import AuthScheme
from ....auth.auth_schemes import AuthSchemeType
from ....auth.auth_schemes import OpenIdConnectWithConfig
from ..common.common import ApiParameter


class OpenIdConfig(BaseModel):
  """Represents OpenID Connect configuration.

  Attributes:
      client_id: The client ID.
      auth_uri: The authorization URI.
      token_uri: The token URI.
      client_secret: The client secret.

  Example:
      config = OpenIdConfig(
          client_id="your_client_id",
          auth_uri="https://accounts.google.com/o/oauth2/auth",
          token_uri="https://oauth2.googleapis.com/token",
          client_secret="your_client_secret",
          redirect
      )
  """

  client_id: str
  auth_uri: str
  token_uri: str
  client_secret: str
  redirect_uri: Optional[str]


def token_to_scheme_credential(
    token_type: Literal["apikey", "oauth2Token"],
    location: Optional[Literal["header", "query", "cookie"]] = None,
    name: Optional[str] = None,
    credential_value: Optional[str] = None,
) -> Tuple[AuthScheme, AuthCredential]:
  """Creates a AuthScheme and AuthCredential for API key or bearer token.

  Examples:
  ```
  # API Key in header
  auth_scheme, auth_credential = token_to_scheme_credential("apikey", "header",
  "X-API-Key", "your_api_key_value")

  # API Key in query parameter
  auth_scheme, auth_credential = token_to_scheme_credential("apikey", "query",
  "api_key", "your_api_key_value")

  # OAuth2 Bearer Token in Authorization header
  auth_scheme, auth_credential = token_to_scheme_credential("oauth2Token",
  "header", "Authorization", "your_bearer_token_value")
  ```

  Args:
      type: 'apikey' or 'oauth2Token'.
      location: 'header', 'query', or 'cookie' (only 'header' for oauth2Token).
      name: The name of the header, query parameter, or cookie.
      credential_value:  The value of the API Key/ Token.

  Returns:
      Tuple: (AuthScheme, AuthCredential)

  Raises:
      ValueError: For invalid type or location.
  """
  if token_type == "apikey":
    in_: APIKeyIn
    if location == "header":
      in_ = APIKeyIn.header
    elif location == "query":
      in_ = APIKeyIn.query
    elif location == "cookie":
      in_ = APIKeyIn.cookie
    else:
      raise ValueError(f"Invalid location for apiKey: {location}")
    auth_scheme = APIKey(**{
        "type": AuthSchemeType.apiKey,
        "in": in_,
        "name": name,
    })
    if credential_value:
      auth_credential = AuthCredential(
          auth_type=AuthCredentialTypes.API_KEY, api_key=credential_value
      )
    else:
      auth_credential = None

    return auth_scheme, auth_credential

  elif token_type == "oauth2Token":
    # ignore location. OAuth2 Bearer Token is always in Authorization header.
    auth_scheme = HTTPBearer(
        bearerFormat="JWT"
    )  # Common format, can be omitted.
    if credential_value:
      auth_credential = AuthCredential(
          auth_type=AuthCredentialTypes.HTTP,
          http=HttpAuth(
              scheme="bearer",
              credentials=HttpCredentials(token=credential_value),
          ),
      )
    else:
      auth_credential = None

    return auth_scheme, auth_credential

  else:
    raise ValueError(f"Invalid security scheme type: {type}")


def service_account_dict_to_scheme_credential(
    config: Dict[str, Any],
    scopes: List[str],
) -> Tuple[AuthScheme, AuthCredential]:
  """Creates AuthScheme and AuthCredential for Google Service Account.

  Returns a bearer token scheme, and a service account credential.

  Args:
      config: A ServiceAccount object containing the Google Service Account
        configuration.
      scopes: A list of scopes to be used.

  Returns:
      Tuple: (AuthScheme, AuthCredential)
  """
  auth_scheme = HTTPBearer(bearerFormat="JWT")
  service_account = ServiceAccount(
      service_account_credential=ServiceAccountCredential.model_construct(
          **config
      ),
      scopes=scopes,
  )
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=service_account,
  )
  return auth_scheme, auth_credential


def service_account_scheme_credential(
    config: ServiceAccount,
) -> Tuple[AuthScheme, AuthCredential]:
  """Creates AuthScheme and AuthCredential for Google Service Account.

  Returns a bearer token scheme, and a service account credential.

  Args:
      config: A ServiceAccount object containing the Google Service Account
        configuration.

  Returns:
      Tuple: (AuthScheme, AuthCredential)
  """
  auth_scheme = HTTPBearer(bearerFormat="JWT")
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT, service_account=config
  )
  return auth_scheme, auth_credential


def openid_dict_to_scheme_credential(
    config_dict: Dict[str, Any],
    scopes: List[str],
    credential_dict: Dict[str, Any],
) -> Tuple[OpenIdConnectWithConfig, AuthCredential]:
  """Constructs OpenID scheme and credential from configuration and credential dictionaries.

  Args:
      config_dict: Dictionary containing OpenID Connect configuration,  must
        include at least 'authorization_endpoint' and 'token_endpoint'.
      scopes: List of scopes to be used.
      credential_dict: Dictionary containing credential information, must
        include 'client_id', 'client_secret', and 'scopes'.  May optionally
        include 'redirect_uri'.

  Returns:
      Tuple: (OpenIdConnectWithConfig, AuthCredential)

  Raises:
      ValueError: If required fields are missing in the input dictionaries.
  """

  # Validate and create the OpenIdConnectWithConfig scheme
  try:
    config_dict["scopes"] = scopes
    # If user provides the OpenID Config as a static dict, it may not contain
    # openIdConnect URL.
    if "openIdConnectUrl" not in config_dict:
      config_dict["openIdConnectUrl"] = ""
    openid_scheme = OpenIdConnectWithConfig.model_validate(config_dict)
  except ValidationError as e:
    raise ValueError(f"Invalid OpenID Connect configuration: {e}") from e

  # Attempt to adjust credential_dict if this is a key downloaded from Google
  # OAuth config
  if len(list(credential_dict.values())) == 1:
    credential_value = list(credential_dict.values())[0]
    if "client_id" in credential_value and "client_secret" in credential_value:
      credential_dict = credential_value

  # Validate credential_dict
  required_credential_fields = ["client_id", "client_secret"]
  missing_fields = [
      field
      for field in required_credential_fields
      if field not in credential_dict
  ]
  if missing_fields:
    raise ValueError(
        "Missing required fields in credential_dict:"
        f" {', '.join(missing_fields)}"
    )

  # Construct AuthCredential
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id=credential_dict["client_id"],
          client_secret=credential_dict["client_secret"],
          redirect_uri=credential_dict.get("redirect_uri", None),
      ),
  )

  return openid_scheme, auth_credential


def openid_url_to_scheme_credential(
    openid_url: str, scopes: List[str], credential_dict: Dict[str, Any]
) -> Tuple[OpenIdConnectWithConfig, AuthCredential]:
  """Constructs OpenID scheme and credential from OpenID URL, scopes, and credential dictionary.

  Fetches OpenID configuration from the provided URL.

  Args:
      openid_url: The OpenID Connect discovery URL.
      scopes: List of scopes to be used.
      credential_dict: Dictionary containing credential information, must
        include at least "client_id" and "client_secret", may optionally include
        "redirect_uri" and "scope"

  Returns:
      Tuple: (AuthScheme, AuthCredential)

  Raises:
      ValueError: If the OpenID URL is invalid, fetching fails, or required
        fields are missing.
      requests.exceptions.RequestException:  If there's an error during the
          HTTP request.
  """
  try:
    response = requests.get(openid_url, timeout=10)
    response.raise_for_status()
    config_dict = response.json()
  except requests.exceptions.RequestException as e:
    raise ValueError(
        f"Failed to fetch OpenID configuration from {openid_url}: {e}"
    ) from e
  except ValueError as e:
    raise ValueError(
        "Invalid JSON response from OpenID configuration endpoint"
        f" {openid_url}: {e}"
    ) from e

  # Add openIdConnectUrl to config dict
  config_dict["openIdConnectUrl"] = openid_url

  return openid_dict_to_scheme_credential(config_dict, scopes, credential_dict)


INTERNAL_AUTH_PREFIX = "_auth_prefix_vaf_"


def credential_to_param(
    auth_scheme: AuthScheme,
    auth_credential: AuthCredential,
) -> Tuple[Optional[ApiParameter], Optional[Dict[str, Any]]]:
  """Converts AuthCredential and AuthScheme to a Parameter and a dictionary for additional kwargs.

  This function now supports all credential types returned by the exchangers:
  - API Key
  - HTTP Bearer (for Bearer tokens, OAuth2, Service Account, OpenID Connect)
  - OAuth2 and OpenID Connect (returns None, None, as the token is now a Bearer
  token)
  - Service Account (returns None, None, as the token is now a Bearer token)

  Args:
      auth_scheme: The AuthScheme object.
      auth_credential: The AuthCredential object.

  Returns:
      Tuple: (ApiParameter, Dict[str, Any])
  """
  if not auth_credential:
    return None, None

  if (
      auth_scheme.type_ == AuthSchemeType.apiKey
      and auth_credential
      and auth_credential.api_key
  ):
    param_name = auth_scheme.name or ""
    python_name = INTERNAL_AUTH_PREFIX + param_name
    if auth_scheme.in_ == APIKeyIn.header:
      param_location = "header"
    elif auth_scheme.in_ == APIKeyIn.query:
      param_location = "query"
    elif auth_scheme.in_ == APIKeyIn.cookie:
      param_location = "cookie"
    else:
      raise ValueError(f"Invalid API Key location: {auth_scheme.in_}")

    param = ApiParameter(
        original_name=param_name,
        param_location=param_location,
        param_schema=Schema(type="string"),
        description=auth_scheme.description or "",
        py_name=python_name,
    )
    kwargs = {param.py_name: auth_credential.api_key}
    return param, kwargs

  # TODO(cheliu): Split handling for OpenIDConnect scheme and native HTTPBearer
  # Scheme
  elif (
      auth_credential and auth_credential.auth_type == AuthCredentialTypes.HTTP
  ):
    if (
        auth_credential
        and auth_credential.http
        and auth_credential.http.credentials
        and auth_credential.http.credentials.token
    ):
      param = ApiParameter(
          original_name="Authorization",
          param_location="header",
          param_schema=Schema(type="string"),
          description=auth_scheme.description or "Bearer token",
          py_name=INTERNAL_AUTH_PREFIX + "Authorization",
      )
      kwargs = {
          param.py_name: f"Bearer {auth_credential.http.credentials.token}"
      }
      return param, kwargs
    elif (
        auth_credential
        and auth_credential.http
        and auth_credential.http.credentials
        and (
            auth_credential.http.credentials.username
            or auth_credential.http.credentials.password
        )
    ):
      # Basic Auth is explicitly NOT supported
      raise NotImplementedError("Basic Authentication is not supported.")
    else:
      raise ValueError("Invalid HTTP auth credentials")

  # Service Account tokens, OAuth2 Tokens and OpenID Tokens are now handled as
  # Bearer tokens.
  elif (auth_scheme.type_ == AuthSchemeType.oauth2 and auth_credential) or (
      auth_scheme.type_ == AuthSchemeType.openIdConnect and auth_credential
  ):
    if (
        auth_credential.http
        and auth_credential.http.credentials
        and auth_credential.http.credentials.token
    ):
      param = ApiParameter(
          original_name="Authorization",
          param_location="header",
          param_schema=Schema(type="string"),
          description=auth_scheme.description or "Bearer token",
          py_name=INTERNAL_AUTH_PREFIX + "Authorization",
      )
      kwargs = {
          param.py_name: f"Bearer {auth_credential.http.credentials.token}"
      }
      return param, kwargs
    return None, None
  else:
    raise ValueError("Invalid security scheme and credential combination")


def dict_to_auth_scheme(data: Dict[str, Any]) -> AuthScheme:
  """Converts a dictionary to a FastAPI AuthScheme object.

  Args:
      data: The dictionary representing the security scheme.

  Returns:
      A AuthScheme object (APIKey, HTTPBase, OAuth2, OpenIdConnect, or
      HTTPBearer).

  Raises:
      ValueError: If the 'type' field is missing or invalid, or if the
          dictionary cannot be converted to the corresponding Pydantic model.

  Example:
  ```python
  api_key_data = {
      "type": "apiKey",
      "in": "header",
      "name": "X-API-Key",
  }
  api_key_scheme = dict_to_auth_scheme(api_key_data)

  bearer_data = {
      "type": "http",
      "scheme": "bearer",
      "bearerFormat": "JWT",
  }
  bearer_scheme = dict_to_auth_scheme(bearer_data)


  oauth2_data = {
      "type": "oauth2",
      "flows": {
          "authorizationCode": {
              "authorizationUrl": "https://example.com/auth",
              "tokenUrl": "https://example.com/token",
          }
      }
  }
  oauth2_scheme = dict_to_auth_scheme(oauth2_data)

  openid_data = {
      "type": "openIdConnect",
      "openIdConnectUrl": "https://example.com/.well-known/openid-configuration"
  }
  openid_scheme = dict_to_auth_scheme(openid_data)


  ```
  """
  if "type" not in data:
    raise ValueError("Missing 'type' field in security scheme dictionary.")

  security_type = data["type"]
  try:
    if security_type == "apiKey":
      return APIKey.model_validate(data)
    elif security_type == "http":
      if data.get("scheme") == "bearer":
        return HTTPBearer.model_validate(data)
      else:
        return HTTPBase.model_validate(data)  # Generic HTTP
    elif security_type == "oauth2":
      return OAuth2.model_validate(data)
    elif security_type == "openIdConnect":
      return OpenIdConnect.model_validate(data)
    else:
      raise ValueError(f"Invalid security scheme type: {security_type}")

  except ValidationError as e:
    raise ValueError(f"Invalid security scheme data: {e}") from e



================================================
FILE: src/google/adk/tools/openapi_tool/auth/credential_exchangers/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .auto_auth_credential_exchanger import AutoAuthCredentialExchanger
from .base_credential_exchanger import BaseAuthCredentialExchanger
from .oauth2_exchanger import OAuth2CredentialExchanger
from .service_account_exchanger import ServiceAccountCredentialExchanger

__all__ = [
    'AutoAuthCredentialExchanger',
    'BaseAuthCredentialExchanger',
    'OAuth2CredentialExchanger',
    'ServiceAccountCredentialExchanger',
]



================================================
FILE: src/google/adk/tools/openapi_tool/auth/credential_exchangers/auto_auth_credential_exchanger.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict
from typing import Optional
from typing import Type

from .....auth.auth_credential import AuthCredential
from .....auth.auth_credential import AuthCredentialTypes
from .....auth.auth_schemes import AuthScheme
from .base_credential_exchanger import BaseAuthCredentialExchanger
from .oauth2_exchanger import OAuth2CredentialExchanger
from .service_account_exchanger import ServiceAccountCredentialExchanger


class AutoAuthCredentialExchanger(BaseAuthCredentialExchanger):
  """Automatically selects the appropriate credential exchanger based on the auth scheme.

  Optionally, an override can be provided to use a specific exchanger for a
  given auth scheme.

  Example (common case):
  ```
  exchanger = AutoAuthCredentialExchanger()
  auth_credential = exchanger.exchange_credential(
      auth_scheme=service_account_scheme,
      auth_credential=service_account_credential,
  )
  # Returns an oauth token in the form of a bearer token.
  ```

  Example (use CustomAuthExchanger for OAuth2):
  ```
  exchanger = AutoAuthCredentialExchanger(
      custom_exchangers={
          AuthScheme.OAUTH2: CustomAuthExchanger,
      }
  )
  ```

  Attributes:
    exchangers: A dictionary mapping auth scheme to credential exchanger class.
  """

  def __init__(
      self,
      custom_exchangers: Optional[
          Dict[str, Type[BaseAuthCredentialExchanger]]
      ] = None,
  ):
    """Initializes the AutoAuthCredentialExchanger.

    Args:
      custom_exchangers: Optional dictionary for adding or overriding auth
        exchangers. The key is the auth scheme, and the value is the credential
        exchanger class.
    """
    self.exchangers = {
        AuthCredentialTypes.OAUTH2: OAuth2CredentialExchanger,
        AuthCredentialTypes.OPEN_ID_CONNECT: OAuth2CredentialExchanger,
        AuthCredentialTypes.SERVICE_ACCOUNT: ServiceAccountCredentialExchanger,
    }

    if custom_exchangers:
      self.exchangers.update(custom_exchangers)

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> Optional[AuthCredential]:
    """Automatically exchanges for the credential uses the appropriate credential exchanger.

    Args:
        auth_scheme (AuthScheme): The security scheme.
        auth_credential (AuthCredential): Optional. The authentication
          credential.

    Returns: (AuthCredential)
        A new AuthCredential object containing the exchanged credential.

    """
    if not auth_credential:
      return None

    exchanger_class = self.exchangers.get(
        auth_credential.auth_type if auth_credential else None
    )

    if not exchanger_class:
      return auth_credential

    exchanger = exchanger_class()
    return exchanger.exchange_credential(auth_scheme, auth_credential)



================================================
FILE: src/google/adk/tools/openapi_tool/auth/credential_exchangers/base_credential_exchanger.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Optional

from .....auth.auth_credential import (
    AuthCredential,
)
from .....auth.auth_schemes import AuthScheme


class AuthCredentialMissingError(Exception):
  """Exception raised when required authentication credentials are missing."""

  def __init__(self, message: str):
    super().__init__(message)
    self.message = message


class BaseAuthCredentialExchanger:
  """Base class for authentication credential exchangers."""

  @abc.abstractmethod
  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Exchanges the provided authentication credential for a usable token/credential.

    Args:
        auth_scheme: The security scheme.
        auth_credential: The authentication credential.

    Returns:
        An updated AuthCredential object containing the fetched credential.
        For simple schemes like API key, it may return the original credential
        if no exchange is needed.

    Raises:
        NotImplementedError: If the method is not implemented by a subclass.
    """
    raise NotImplementedError("Subclasses must implement exchange_credential.")



================================================
FILE: src/google/adk/tools/openapi_tool/auth/credential_exchangers/oauth2_exchanger.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Credential fetcher for OpenID Connect."""

from typing import Optional

from .....auth.auth_credential import AuthCredential
from .....auth.auth_credential import AuthCredentialTypes
from .....auth.auth_credential import HttpAuth
from .....auth.auth_credential import HttpCredentials
from .....auth.auth_schemes import AuthScheme
from .....auth.auth_schemes import AuthSchemeType
from .base_credential_exchanger import BaseAuthCredentialExchanger


class OAuth2CredentialExchanger(BaseAuthCredentialExchanger):
  """Fetches credentials for OAuth2 and OpenID Connect."""

  def _check_scheme_credential_type(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ):
    if not auth_credential:
      raise ValueError(
          "auth_credential is empty. Please create AuthCredential using"
          " OAuth2Auth."
      )

    if auth_scheme.type_ not in (
        AuthSchemeType.openIdConnect,
        AuthSchemeType.oauth2,
    ):
      raise ValueError(
          "Invalid security scheme, expect AuthSchemeType.openIdConnect or "
          f"AuthSchemeType.oauth2 auth scheme, but got {auth_scheme.type_}"
      )

    if not auth_credential.oauth2 and not auth_credential.http:
      raise ValueError(
          "auth_credential is not configured with oauth2. Please"
          " create AuthCredential and set OAuth2Auth."
      )

  def generate_auth_token(
      self,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Generates an auth token from the authorization response.

    Args:
        auth_scheme: The OpenID Connect or OAuth2 auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential object containing the HTTP bearer access token. If the
        HTTP bearer token cannot be generated, return the original credential.
    """

    if not auth_credential.oauth2.access_token:
      return auth_credential

    # Return the access token as a bearer token.
    updated_credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,  # Store as a bearer token
        http=HttpAuth(
            scheme="bearer",
            credentials=HttpCredentials(
                token=auth_credential.oauth2.access_token
            ),
        ),
    )
    return updated_credential

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Exchanges the OpenID Connect auth credential for an access token or an auth URI.

    Args:
        auth_scheme: The auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential object containing the HTTP Bearer access token.

    Raises:
        ValueError: If the auth scheme or auth credential is invalid.
    """
    # TODO(cheliu): Implement token refresh flow

    self._check_scheme_credential_type(auth_scheme, auth_credential)

    # If token is already HTTPBearer token, do nothing assuming that this token
    #  is valid.
    if auth_credential.http:
      return auth_credential

    # If access token is exchanged, exchange a HTTPBearer token.
    if auth_credential.oauth2.access_token:
      return self.generate_auth_token(auth_credential)

    return None



================================================
FILE: src/google/adk/tools/openapi_tool/auth/credential_exchangers/service_account_exchanger.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Credential fetcher for Google Service Account."""

from typing import Optional

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.oauth2.credentials

from .....auth.auth_credential import (
    AuthCredential,
    AuthCredentialTypes,
    HttpAuth,
    HttpCredentials,
)
from .....auth.auth_schemes import AuthScheme
from .base_credential_exchanger import AuthCredentialMissingError, BaseAuthCredentialExchanger


class ServiceAccountCredentialExchanger(BaseAuthCredentialExchanger):
  """Fetches credentials for Google Service Account.

  Uses the default service credential if `use_default_credential = True`.
  Otherwise, uses the service account credential provided in the auth
  credential.
  """

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Exchanges the service account auth credential for an access token.

    If auth_credential contains a service account credential, it will be used
    to fetch an access token. Otherwise, the default service credential will be
    used for fetching an access token.

    Args:
        auth_scheme: The auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential in HTTPBearer format, containing the access token.
    """
    if (
        auth_credential is None
        or auth_credential.service_account is None
        or (
            auth_credential.service_account.service_account_credential is None
            and not auth_credential.service_account.use_default_credential
        )
    ):
      raise AuthCredentialMissingError(
          "Service account credentials are missing. Please provide them, or set"
          " `use_default_credential = True` to use application default"
          " credential in a hosted service like Cloud Run."
      )

    try:
      if auth_credential.service_account.use_default_credential:
        credentials, _ = google.auth.default()
      else:
        config = auth_credential.service_account
        credentials = service_account.Credentials.from_service_account_info(
            config.service_account_credential.model_dump(), scopes=config.scopes
        )

      credentials.refresh(Request())

      updated_credential = AuthCredential(
          auth_type=AuthCredentialTypes.HTTP,  # Store as a bearer token
          http=HttpAuth(
              scheme="bearer",
              credentials=HttpCredentials(token=credentials.token),
          ),
      )
      return updated_credential

    except Exception as e:
      raise AuthCredentialMissingError(
          f"Failed to exchange service account token: {e}"
      ) from e



================================================
FILE: src/google/adk/tools/openapi_tool/common/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import common

__all__ = [
    'common',
]



================================================
FILE: src/google/adk/tools/openapi_tool/common/common.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keyword
import re
from typing import Any, Dict, List, Optional, Union

from fastapi.openapi.models import Response
from fastapi.openapi.models import Schema
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_serializer


def to_snake_case(text: str) -> str:
  """Converts a string into snake_case.

  Handles lowerCamelCase, UpperCamelCase, or space-separated case, acronyms
  (e.g., "REST API") and consecutive uppercase letters correctly.  Also handles
  mixed cases with and without spaces.

  Examples:
  ```
  to_snake_case('camelCase') -> 'camel_case'
  to_snake_case('UpperCamelCase') -> 'upper_camel_case'
  to_snake_case('space separated') -> 'space_separated'
  ```

  Args:
      text: The input string.

  Returns:
      The snake_case version of the string.
  """

  # Handle spaces and non-alphanumeric characters (replace with underscores)
  text = re.sub(r'[^a-zA-Z0-9]+', '_', text)

  # Insert underscores before uppercase letters (handling both CamelCases)
  text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)  # lowerCamelCase
  text = re.sub(
      r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text
  )  # UpperCamelCase and acronyms

  # Convert to lowercase
  text = text.lower()

  # Remove consecutive underscores (clean up extra underscores)
  text = re.sub(r'_+', '_', text)

  # Remove leading and trailing underscores
  text = text.strip('_')

  return text


def rename_python_keywords(s: str, prefix: str = 'param_') -> str:
  """Renames Python keywords by adding a prefix.

  Example:
  ```
  rename_python_keywords('if') -> 'param_if'
  rename_python_keywords('for') -> 'param_for'
  ```

  Args:
      s: The input string.
      prefix: The prefix to add to the keyword.

  Returns:
      The renamed string.
  """
  if keyword.iskeyword(s):
    return prefix + s
  return s


class ApiParameter(BaseModel):
  """Data class representing a function parameter."""

  original_name: str
  param_location: str
  param_schema: Union[str, Schema]
  description: Optional[str] = ''
  py_name: Optional[str] = ''
  type_value: type[Any] = Field(default=None, init_var=False)
  type_hint: str = Field(default=None, init_var=False)
  required: bool = False

  def model_post_init(self, _: Any):
    self.py_name = (
        self.py_name
        if self.py_name
        else rename_python_keywords(to_snake_case(self.original_name))
    )
    if isinstance(self.param_schema, str):
      self.param_schema = Schema.model_validate_json(self.param_schema)

    self.description = self.description or self.param_schema.description or ''
    self.type_value = TypeHintHelper.get_type_value(self.param_schema)
    self.type_hint = TypeHintHelper.get_type_hint(self.param_schema)
    return self

  @model_serializer
  def _serialize(self):
    return {
        'original_name': self.original_name,
        'param_location': self.param_location,
        'param_schema': self.param_schema,
        'description': self.description,
        'py_name': self.py_name,
    }

  def __str__(self):
    return f'{self.py_name}: {self.type_hint}'

  def to_arg_string(self):
    """Converts the parameter to an argument string for function call."""
    return f'{self.py_name}={self.py_name}'

  def to_dict_property(self):
    """Converts the parameter to a key:value string for dict property."""
    return f'"{self.py_name}": {self.py_name}'

  def to_pydoc_string(self):
    """Converts the parameter to a PyDoc parameter docstr."""
    return PydocHelper.generate_param_doc(self)


class TypeHintHelper:
  """Helper class for generating type hints."""

  @staticmethod
  def get_type_value(schema: Schema) -> Any:
    """Generates the Python type value for a given parameter."""
    param_type = schema.type if schema.type else Any

    if param_type == 'integer':
      return int
    elif param_type == 'number':
      return float
    elif param_type == 'boolean':
      return bool
    elif param_type == 'string':
      return str
    elif param_type == 'array':
      items_type = Any
      if schema.items and schema.items.type:
        items_type = schema.items.type

      if items_type == 'object':
        return List[Dict[str, Any]]
      else:
        type_map = {
            'integer': int,
            'number': float,
            'boolean': bool,
            'string': str,
            'object': Dict[str, Any],
            'array': List[Any],
        }
        return List[type_map.get(items_type, 'Any')]
    elif param_type == 'object':
      return Dict[str, Any]
    else:
      return Any

  @staticmethod
  def get_type_hint(schema: Schema) -> str:
    """Generates the Python type in string for a given parameter."""
    param_type = schema.type if schema.type else 'Any'

    if param_type == 'integer':
      return 'int'
    elif param_type == 'number':
      return 'float'
    elif param_type == 'boolean':
      return 'bool'
    elif param_type == 'string':
      return 'str'
    elif param_type == 'array':
      items_type = 'Any'
      if schema.items and schema.items.type:
        items_type = schema.items.type

      if items_type == 'object':
        return 'List[Dict[str, Any]]'
      else:
        type_map = {
            'integer': 'int',
            'number': 'float',
            'boolean': 'bool',
            'string': 'str',
        }
        return f"List[{type_map.get(items_type, 'Any')}]"
    elif param_type == 'object':
      return 'Dict[str, Any]'
    else:
      return 'Any'


class PydocHelper:
  """Helper class for generating PyDoc strings."""

  @staticmethod
  def generate_param_doc(
      param: ApiParameter,
  ) -> str:
    """Generates a parameter documentation string.

    Args:
      param: ApiParameter - The parameter to generate the documentation for.

    Returns:
      str: The generated parameter Python documentation string.
    """
    description = param.description.strip() if param.description else ''
    param_doc = f'{param.py_name} ({param.type_hint}): {description}'

    if param.param_schema.type == 'object':
      properties = param.param_schema.properties
      if properties:
        param_doc += ' Object properties:\n'
        for prop_name, prop_details in properties.items():
          prop_desc = prop_details.description or ''
          prop_type = TypeHintHelper.get_type_hint(prop_details)
          param_doc += f'       {prop_name} ({prop_type}): {prop_desc}\n'

    return param_doc

  @staticmethod
  def generate_return_doc(responses: Dict[str, Response]) -> str:
    """Generates a return value documentation string.

    Args:
      responses: Dict[str, TypedDict[Response]] - Response in an OpenAPI
        Operation

    Returns:
      str: The generated return value Python documentation string.
    """
    return_doc = ''

    # Only consider 2xx responses for return type hinting.
    # Returns the 2xx response with the smallest status code number and with
    # content defined.
    sorted_responses = sorted(responses.items(), key=lambda item: int(item[0]))
    qualified_response = next(
        filter(
            lambda r: r[0].startswith('2') and r[1].content,
            sorted_responses,
        ),
        None,
    )
    if not qualified_response:
      return ''
    response_details = qualified_response[1]

    description = (response_details.description or '').strip()
    content = response_details.content or {}

    # Generate return type hint and properties for the first response type.
    # TODO(cheliu): Handle multiple content types.
    for _, schema_details in content.items():
      schema = schema_details.schema_ or {}

      # Use a dummy Parameter object for return type hinting.
      dummy_param = ApiParameter(
          original_name='', param_location='', param_schema=schema
      )
      return_doc = f'Returns ({dummy_param.type_hint}): {description}'

      response_type = schema.type or 'Any'
      if response_type != 'object':
        break
      properties = schema.properties
      if not properties:
        break
      return_doc += ' Object properties:\n'
      for prop_name, prop_details in properties.items():
        prop_desc = prop_details.description or ''
        prop_type = TypeHintHelper.get_type_hint(prop_details)
        return_doc += f'        {prop_name} ({prop_type}): {prop_desc}\n'
      break

    return return_doc



================================================
FILE: src/google/adk/tools/openapi_tool/openapi_spec_parser/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .openapi_spec_parser import OpenApiSpecParser, OperationEndpoint, ParsedOperation
from .openapi_toolset import OpenAPIToolset
from .operation_parser import OperationParser
from .rest_api_tool import AuthPreparationState, RestApiTool, snake_to_lower_camel, to_gemini_schema
from .tool_auth_handler import ToolAuthHandler

__all__ = [
    'OpenApiSpecParser',
    'OperationEndpoint',
    'ParsedOperation',
    'OpenAPIToolset',
    'OperationParser',
    'RestApiTool',
    'to_gemini_schema',
    'snake_to_lower_camel',
    'AuthPreparationState',
    'ToolAuthHandler',
]



================================================
FILE: src/google/adk/tools/openapi_tool/openapi_spec_parser/openapi_spec_parser.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from fastapi.openapi.models import Operation
from pydantic import BaseModel

from ....auth.auth_credential import AuthCredential
from ....auth.auth_schemes import AuthScheme
from ..common.common import ApiParameter
from ..common.common import to_snake_case
from .operation_parser import OperationParser


class OperationEndpoint(BaseModel):
  base_url: str
  path: str
  method: str


class ParsedOperation(BaseModel):
  name: str
  description: str
  endpoint: OperationEndpoint
  operation: Operation
  parameters: List[ApiParameter]
  return_value: ApiParameter
  auth_scheme: Optional[AuthScheme] = None
  auth_credential: Optional[AuthCredential] = None
  additional_context: Optional[Any] = None


class OpenApiSpecParser:
  """Generates Python code, JSON schema, and callables for an OpenAPI operation.

  This class takes an OpenApiOperation object and provides methods to generate:
  1. A string representation of a Python function that handles the operation.
  2. A JSON schema representing the input parameters of the operation.
  3. A callable Python object (a function) that can execute the operation.
  """

  def parse(self, openapi_spec_dict: Dict[str, Any]) -> List[ParsedOperation]:
    """Extracts an OpenAPI spec dict into a list of ParsedOperation objects.

    ParsedOperation objects are further used for generating RestApiTool.

    Args:
        openapi_spec_dict: A dictionary representing the OpenAPI specification.

    Returns:
        A list of ParsedOperation objects.
    """

    openapi_spec_dict = self._resolve_references(openapi_spec_dict)
    operations = self._collect_operations(openapi_spec_dict)
    return operations

  def _collect_operations(
      self, openapi_spec: Dict[str, Any]
  ) -> List[ParsedOperation]:
    """Collects operations from an OpenAPI spec."""
    operations = []

    # Taking first server url, or default to empty string if not present
    base_url = ""
    if openapi_spec.get("servers"):
      base_url = openapi_spec["servers"][0].get("url", "")

    # Get global security scheme (if any)
    global_scheme_name = None
    if openapi_spec.get("security"):
      # Use first scheme by default.
      scheme_names = list(openapi_spec["security"][0].keys())
      global_scheme_name = scheme_names[0] if scheme_names else None

    auth_schemes = openapi_spec.get("components", {}).get("securitySchemes", {})

    for path, path_item in openapi_spec.get("paths", {}).items():
      if path_item is None:
        continue

      for method in (
          "get",
          "post",
          "put",
          "delete",
          "patch",
          "head",
          "options",
          "trace",
      ):
        operation_dict = path_item.get(method)
        if operation_dict is None:
          continue

        # If operation ID is missing, assign an operation id based on path
        # and method
        if "operationId" not in operation_dict:
          temp_id = to_snake_case(f"{path}_{method}")
          operation_dict["operationId"] = temp_id

        url = OperationEndpoint(base_url=base_url, path=path, method=method)
        operation = Operation.model_validate(operation_dict)
        operation_parser = OperationParser(operation)

        # Check for operation-specific auth scheme
        auth_scheme_name = operation_parser.get_auth_scheme_name()
        auth_scheme_name = (
            auth_scheme_name if auth_scheme_name else global_scheme_name
        )
        auth_scheme = (
            auth_schemes.get(auth_scheme_name) if auth_scheme_name else None
        )

        parsed_op = ParsedOperation(
            name=operation_parser.get_function_name(),
            description=operation.description or operation.summary or "",
            endpoint=url,
            operation=operation,
            parameters=operation_parser.get_parameters(),
            return_value=operation_parser.get_return_value(),
            auth_scheme=auth_scheme,
            auth_credential=None,  # Placeholder
            additional_context={},
        )
        operations.append(parsed_op)

    return operations

  def _resolve_references(self, openapi_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolves all $ref references in an OpenAPI specification.

    Handles circular references correctly.

    Args:
        openapi_spec: A dictionary representing the OpenAPI specification.

    Returns:
        A dictionary representing the OpenAPI specification with all references
        resolved.
    """

    openapi_spec = copy.deepcopy(openapi_spec)  # Work on a copy
    resolved_cache = {}  # Cache resolved references

    def resolve_ref(ref_string, current_doc):
      """Resolves a single $ref string."""
      parts = ref_string.split("/")
      if parts[0] != "#":
        raise ValueError(f"External references not supported: {ref_string}")

      current = current_doc
      for part in parts[1:]:
        if part in current:
          current = current[part]
        else:
          return None  # Reference not found
      return current

    def recursive_resolve(obj, current_doc, seen_refs=None):
      """Recursively resolves references, handling circularity.

      Args:
          obj: The object to traverse.
          current_doc:  Document to search for refs.
          seen_refs: A set to track already-visited references (for circularity
            detection).

      Returns:
          The resolved object.
      """
      if seen_refs is None:
        seen_refs = set()  # Initialize the set if it's the first call

      if isinstance(obj, dict):
        if "$ref" in obj and isinstance(obj["$ref"], str):
          ref_string = obj["$ref"]

          # Check for circularity
          if ref_string in seen_refs and ref_string not in resolved_cache:
            # Circular reference detected! Return a *copy* of the object,
            # but *without* the $ref.  This breaks the cycle while
            # still maintaining the overall structure.
            return {k: v for k, v in obj.items() if k != "$ref"}

          seen_refs.add(ref_string)  # Add the reference to the set

          # Check if we have a cached resolved value
          if ref_string in resolved_cache:
            return copy.deepcopy(resolved_cache[ref_string])

          resolved_value = resolve_ref(ref_string, current_doc)
          if resolved_value is not None:
            # Recursively resolve the *resolved* value,
            # passing along the 'seen_refs' set
            resolved_value = recursive_resolve(
                resolved_value, current_doc, seen_refs
            )
            resolved_cache[ref_string] = resolved_value
            return copy.deepcopy(resolved_value)  # return the cached result
          else:
            return obj  # return original if no resolved value.

        else:
          new_dict = {}
          for key, value in obj.items():
            new_dict[key] = recursive_resolve(value, current_doc, seen_refs)
          return new_dict

      elif isinstance(obj, list):
        return [recursive_resolve(item, current_doc, seen_refs) for item in obj]
      else:
        return obj

    return recursive_resolve(openapi_spec, openapi_spec)



================================================
FILE: src/google/adk/tools/openapi_tool/openapi_spec_parser/openapi_toolset.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Any
from typing import Dict
from typing import Final
from typing import List
from typing import Literal
from typing import Optional

import yaml

from ....auth.auth_credential import AuthCredential
from ....auth.auth_schemes import AuthScheme
from .openapi_spec_parser import OpenApiSpecParser
from .rest_api_tool import RestApiTool

logger = logging.getLogger(__name__)


class OpenAPIToolset:
  """Class for parsing OpenAPI spec into a list of RestApiTool.

  Usage:
  ```
    # Initialize OpenAPI toolset from a spec string.
    openapi_toolset = OpenAPIToolset(spec_str=openapi_spec_str,
      spec_str_type="json")
    # Or, initialize OpenAPI toolset from a spec dictionary.
    openapi_toolset = OpenAPIToolset(spec_dict=openapi_spec_dict)

    # Add all tools to an agent.
    agent = Agent(
      tools=[*openapi_toolset.get_tools()]
    )
    # Or, add a single tool to an agent.
    agent = Agent(
      tools=[openapi_toolset.get_tool('tool_name')]
    )
  ```
  """

  def __init__(
      self,
      *,
      spec_dict: Optional[Dict[str, Any]] = None,
      spec_str: Optional[str] = None,
      spec_str_type: Literal["json", "yaml"] = "json",
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
  ):
    """Initializes the OpenAPIToolset.

    Usage:
    ```
      # Initialize OpenAPI toolset from a spec string.
      openapi_toolset = OpenAPIToolset(spec_str=openapi_spec_str,
        spec_str_type="json")
      # Or, initialize OpenAPI toolset from a spec dictionary.
      openapi_toolset = OpenAPIToolset(spec_dict=openapi_spec_dict)

      # Add all tools to an agent.
      agent = Agent(
        tools=[*openapi_toolset.get_tools()]
      )
      # Or, add a single tool to an agent.
      agent = Agent(
        tools=[openapi_toolset.get_tool('tool_name')]
      )
    ```

    Args:
      spec_dict: The OpenAPI spec dictionary. If provided, it will be used
        instead of loading the spec from a string.
      spec_str: The OpenAPI spec string in JSON or YAML format. It will be used
        when spec_dict is not provided.
      spec_str_type: The type of the OpenAPI spec string. Can be "json" or
        "yaml".
      auth_scheme: The auth scheme to use for all tools. Use AuthScheme or use
        helpers in `google.adk.tools.openapi_tool.auth.auth_helpers`
      auth_credential: The auth credential to use for all tools. Use
        AuthCredential or use helpers in
        `google.adk.tools.openapi_tool.auth.auth_helpers`
    """
    if not spec_dict:
      spec_dict = self._load_spec(spec_str, spec_str_type)
    self.tools: Final[List[RestApiTool]] = list(self._parse(spec_dict))
    if auth_scheme or auth_credential:
      self._configure_auth_all(auth_scheme, auth_credential)

  def _configure_auth_all(
      self, auth_scheme: AuthScheme, auth_credential: AuthCredential
  ):
    """Configure auth scheme and credential for all tools."""

    for tool in self.tools:
      if auth_scheme:
        tool.configure_auth_scheme(auth_scheme)
      if auth_credential:
        tool.configure_auth_credential(auth_credential)

  def get_tools(self) -> List[RestApiTool]:
    """Get all tools in the toolset."""
    return self.tools

  def get_tool(self, tool_name: str) -> Optional[RestApiTool]:
    """Get a tool by name."""
    matching_tool = filter(lambda t: t.name == tool_name, self.tools)
    return next(matching_tool, None)

  def _load_spec(
      self, spec_str: str, spec_type: Literal["json", "yaml"]
  ) -> Dict[str, Any]:
    """Loads the OpenAPI spec string into a dictionary."""
    if spec_type == "json":
      return json.loads(spec_str)
    elif spec_type == "yaml":
      return yaml.safe_load(spec_str)
    else:
      raise ValueError(f"Unsupported spec type: {spec_type}")

  def _parse(self, openapi_spec_dict: Dict[str, Any]) -> List[RestApiTool]:
    """Parse OpenAPI spec into a list of RestApiTool."""
    operations = OpenApiSpecParser().parse(openapi_spec_dict)

    tools = []
    for o in operations:
      tool = RestApiTool.from_parsed_operation(o)
      logger.info("Parsed tool: %s", tool.name)
      tools.append(tool)
    return tools



================================================
FILE: src/google/adk/tools/openapi_tool/openapi_spec_parser/operation_parser.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from textwrap import dedent
from typing import Any, Dict, List, Optional, Union

from fastapi.encoders import jsonable_encoder
from fastapi.openapi.models import Operation
from fastapi.openapi.models import Parameter
from fastapi.openapi.models import Schema

from ..common.common import ApiParameter
from ..common.common import PydocHelper
from ..common.common import to_snake_case


class OperationParser:
  """Generates parameters for Python functions from an OpenAPI operation.

  This class processes an OpenApiOperation object and provides helper methods
  to extract information needed to generate Python function declarations,
  docstrings, signatures, and JSON schemas.  It handles parameter processing,
  name deduplication, and type hint generation.
  """

  def __init__(
      self, operation: Union[Operation, Dict[str, Any], str], should_parse=True
  ):
    """Initializes the OperationParser with an OpenApiOperation.

    Args:
        operation: The OpenApiOperation object or a dictionary to process.
        should_parse: Whether to parse the operation during initialization.
    """
    if isinstance(operation, dict):
      self.operation = Operation.model_validate(operation)
    elif isinstance(operation, str):
      self.operation = Operation.model_validate_json(operation)
    else:
      self.operation = operation

    self.params: List[ApiParameter] = []
    self.return_value: Optional[ApiParameter] = None
    if should_parse:
      self._process_operation_parameters()
      self._process_request_body()
      self._process_return_value()
      self._dedupe_param_names()

  @classmethod
  def load(
      cls,
      operation: Union[Operation, Dict[str, Any]],
      params: List[ApiParameter],
      return_value: Optional[ApiParameter] = None,
  ) -> 'OperationParser':
    parser = cls(operation, should_parse=False)
    parser.params = params
    parser.return_value = return_value
    return parser

  def _process_operation_parameters(self):
    """Processes parameters from the OpenAPI operation."""
    parameters = self.operation.parameters or []
    for param in parameters:
      if isinstance(param, Parameter):
        original_name = param.name
        description = param.description or ''
        location = param.in_ or ''
        schema = param.schema_ or {}  # Use schema_ instead of .schema
        schema.description = (
            description if not schema.description else schema.description
        )
        # param.required can be None
        required = param.required if param.required is not None else False

        self.params.append(
            ApiParameter(
                original_name=original_name,
                param_location=location,
                param_schema=schema,
                description=description,
                required=required,
            )
        )

  def _process_request_body(self):
    """Processes the request body from the OpenAPI operation."""
    request_body = self.operation.requestBody
    if not request_body:
      return

    content = request_body.content or {}
    if not content:
      return

    # If request body is an object, expand the properties as parameters
    for _, media_type_object in content.items():
      schema = media_type_object.schema_ or {}
      description = request_body.description or ''

      if schema and schema.type == 'object':
        properties = schema.properties or {}
        for prop_name, prop_details in properties.items():
          self.params.append(
              ApiParameter(
                  original_name=prop_name,
                  param_location='body',
                  param_schema=prop_details,
                  description=prop_details.description,
              )
          )

      elif schema and schema.type == 'array':
        self.params.append(
            ApiParameter(
                original_name='array',
                param_location='body',
                param_schema=schema,
                description=description,
            )
        )
      else:
        self.params.append(
            # Empty name for unnamed body param
            ApiParameter(
                original_name='',
                param_location='body',
                param_schema=schema,
                description=description,
            )
        )
      break  # Process first mime type only

  def _dedupe_param_names(self):
    """Deduplicates parameter names to avoid conflicts."""
    params_cnt = {}
    for param in self.params:
      name = param.py_name
      if name not in params_cnt:
        params_cnt[name] = 0
      else:
        params_cnt[name] += 1
        param.py_name = f'{name}_{params_cnt[name] -1}'

  def _process_return_value(self) -> Parameter:
    """Returns a Parameter object representing the return type."""
    responses = self.operation.responses or {}
    # Default to Any if no 2xx response or if schema is missing
    return_schema = Schema(type='Any')

    # Take the 20x response with the smallest response code.
    valid_codes = list(
        filter(lambda k: k.startswith('2'), list(responses.keys()))
    )
    min_20x_status_code = min(valid_codes) if valid_codes else None

    if min_20x_status_code and responses[min_20x_status_code].content:
      content = responses[min_20x_status_code].content
      for mime_type in content:
        if content[mime_type].schema_:
          return_schema = content[mime_type].schema_
          break

    self.return_value = ApiParameter(
        original_name='',
        param_location='',
        param_schema=return_schema,
    )

  def get_function_name(self) -> str:
    """Returns the generated function name."""
    operation_id = self.operation.operationId
    if not operation_id:
      raise ValueError('Operation ID is missing')
    return to_snake_case(operation_id)[:60]

  def get_return_type_hint(self) -> str:
    """Returns the return type hint string (like 'str', 'int', etc.)."""
    return self.return_value.type_hint

  def get_return_type_value(self) -> Any:
    """Returns the return type value (like str, int, List[str], etc.)."""
    return self.return_value.type_value

  def get_parameters(self) -> List[ApiParameter]:
    """Returns the list of Parameter objects."""
    return self.params

  def get_return_value(self) -> ApiParameter:
    """Returns the list of Parameter objects."""
    return self.return_value

  def get_auth_scheme_name(self) -> str:
    """Returns the name of the auth scheme for this operation from the spec."""
    if self.operation.security:
      scheme_name = list(self.operation.security[0].keys())[0]
      return scheme_name
    return ''

  def get_pydoc_string(self) -> str:
    """Returns the generated PyDoc string."""
    pydoc_params = [param.to_pydoc_string() for param in self.params]
    pydoc_description = (
        self.operation.summary or self.operation.description or ''
    )
    pydoc_return = PydocHelper.generate_return_doc(
        self.operation.responses or {}
    )
    pydoc_arg_list = chr(10).join(
        f'        {param_doc}' for param_doc in pydoc_params
    )
    return dedent(f"""
        \"\"\"{pydoc_description}

        Args:
        {pydoc_arg_list}

        {pydoc_return}
        \"\"\"
            """).strip()

  def get_json_schema(self) -> Dict[str, Any]:
    """Returns the JSON schema for the function arguments."""
    properties = {
        p.py_name: jsonable_encoder(p.param_schema, exclude_none=True)
        for p in self.params
    }
    return {
        'properties': properties,
        'required': [p.py_name for p in self.params if p.required],
        'title': f"{self.operation.operationId or 'unnamed'}_Arguments",
        'type': 'object',
    }

  def get_signature_parameters(self) -> List[inspect.Parameter]:
    """Returns a list of inspect.Parameter objects for the function."""
    return [
        inspect.Parameter(
            param.py_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=param.type_value,
        )
        for param in self.params
    ]

  def get_annotations(self) -> Dict[str, Any]:
    """Returns a dictionary of parameter annotations for the function."""
    annotations = {p.py_name: p.type_value for p in self.params}
    annotations['return'] = self.get_return_type_value()
    return annotations



================================================
FILE: src/google/adk/tools/openapi_tool/openapi_spec_parser/rest_api_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from fastapi.openapi.models import Operation
from google.genai.types import FunctionDeclaration
from google.genai.types import Schema
import requests
from typing_extensions import override

from ....auth.auth_credential import AuthCredential
from ....auth.auth_schemes import AuthScheme
from ....tools.base_tool import BaseTool
from ...tool_context import ToolContext
from ..auth.auth_helpers import credential_to_param
from ..auth.auth_helpers import dict_to_auth_scheme
from ..auth.credential_exchangers.auto_auth_credential_exchanger import AutoAuthCredentialExchanger
from ..common.common import ApiParameter
from ..common.common import to_snake_case
from .openapi_spec_parser import OperationEndpoint
from .openapi_spec_parser import ParsedOperation
from .operation_parser import OperationParser
from .tool_auth_handler import ToolAuthHandler


def snake_to_lower_camel(snake_case_string: str):
  """Converts a snake_case string to a lower_camel_case string.

  Args:
      snake_case_string: The input snake_case string.

  Returns:
      The lower_camel_case string.
  """
  if "_" not in snake_case_string:
    return snake_case_string

  return "".join([
      s.lower() if i == 0 else s.capitalize()
      for i, s in enumerate(snake_case_string.split("_"))
  ])


# TODO: Switch to Gemini `from_json_schema` util when it is released
# in Gemini SDK.
def normalize_json_schema_type(
    json_schema_type: Optional[Union[str, Sequence[str]]],
) -> tuple[Optional[str], bool]:
  """Converts a JSON Schema Type into Gemini Schema type.

  Adopted and modified from Gemini SDK. This gets the first available schema
  type from JSON Schema, and use it to mark Gemini schema type. If JSON Schema
  contains a list of types, the first non null type is used.

  Remove this after switching to Gemini `from_json_schema`.
  """
  if json_schema_type is None:
    return None, False
  if isinstance(json_schema_type, str):
    if json_schema_type == "null":
      return None, True
    return json_schema_type, False

  non_null_types = []
  nullable = False
  # If json schema type is an array, pick the first non null type.
  for type_value in json_schema_type:
    if type_value == "null":
      nullable = True
    else:
      non_null_types.append(type_value)
  non_null_type = non_null_types[0] if non_null_types else None
  return non_null_type, nullable


# TODO: Switch to Gemini `from_json_schema` util when it is released
# in Gemini SDK.
def to_gemini_schema(openapi_schema: Optional[Dict[str, Any]] = None) -> Schema:
  """Converts an OpenAPI schema dictionary to a Gemini Schema object.

  Args:
      openapi_schema: The OpenAPI schema dictionary.

  Returns:
      A Pydantic Schema object.  Returns None if input is None.
      Raises TypeError if input is not a dict.
  """
  if openapi_schema is None:
    return None

  if not isinstance(openapi_schema, dict):
    raise TypeError("openapi_schema must be a dictionary")

  pydantic_schema_data = {}

  # Adding this to force adding a type to an empty dict
  # This avoid "... one_of or any_of must specify a type" error
  if not openapi_schema.get("type"):
    openapi_schema["type"] = "object"

  for key, value in openapi_schema.items():
    snake_case_key = to_snake_case(key)
    # Check if the snake_case_key exists in the Schema model's fields.
    if snake_case_key in Schema.model_fields:
      if snake_case_key in ["title", "default", "format"]:
        # Ignore these fields as Gemini backend doesn't recognize them, and will
        # throw exception if they appear in the schema.
        # Format: properties[expiration].format: only 'enum' and 'date-time' are
        # supported for STRING type
        continue
      elif snake_case_key == "type":
        schema_type, nullable = normalize_json_schema_type(
            openapi_schema.get("type", None)
        )
        # Adding this to force adding a type to an empty dict
        # This avoid "... one_of or any_of must specify a type" error
        pydantic_schema_data["type"] = schema_type if schema_type else "object"
        pydantic_schema_data["type"] = pydantic_schema_data["type"].upper()
        if nullable:
          pydantic_schema_data["nullable"] = True
      elif snake_case_key == "properties" and isinstance(value, dict):
        pydantic_schema_data[snake_case_key] = {
            k: to_gemini_schema(v) for k, v in value.items()
        }
      elif snake_case_key == "items" and isinstance(value, dict):
        pydantic_schema_data[snake_case_key] = to_gemini_schema(value)
      elif snake_case_key == "any_of" and isinstance(value, list):
        pydantic_schema_data[snake_case_key] = [
            to_gemini_schema(item) for item in value
        ]
      # Important:  Handle cases where the OpenAPI schema might contain lists
      # or other structures that need to be recursively processed.
      elif isinstance(value, list) and snake_case_key not in (
          "enum",
          "required",
          "property_ordering",
      ):
        new_list = []
        for item in value:
          if isinstance(item, dict):
            new_list.append(to_gemini_schema(item))
          else:
            new_list.append(item)
        pydantic_schema_data[snake_case_key] = new_list
      elif isinstance(value, dict) and snake_case_key not in ("properties"):
        # Handle dictionary which is neither properties or items
        pydantic_schema_data[snake_case_key] = to_gemini_schema(value)
      else:
        # Simple value assignment (int, str, bool, etc.)
        pydantic_schema_data[snake_case_key] = value

  return Schema(**pydantic_schema_data)


AuthPreparationState = Literal["pending", "done"]


class RestApiTool(BaseTool):
  """A generic tool that interacts with a REST API.

  * Generates request params and body
  * Attaches auth credentials to API call.

  Example:
  ```
    # Each API operation in the spec will be turned into its own tool
    # Name of the tool is the operationId of that operation, in snake case
    operations = OperationGenerator().parse(openapi_spec_dict)
    tool = [RestApiTool.from_parsed_operation(o) for o in operations]
  ```
  """

  def __init__(
      self,
      name: str,
      description: str,
      endpoint: Union[OperationEndpoint, str],
      operation: Union[Operation, str],
      auth_scheme: Optional[Union[AuthScheme, str]] = None,
      auth_credential: Optional[Union[AuthCredential, str]] = None,
      should_parse_operation=True,
  ):
    """Initializes the RestApiTool with the given parameters.

    To generate RestApiTool from OpenAPI Specs, use OperationGenerator.
    Example:
    ```
      # Each API operation in the spec will be turned into its own tool
      # Name of the tool is the operationId of that operation, in snake case
      operations = OperationGenerator().parse(openapi_spec_dict)
      tool = [RestApiTool.from_parsed_operation(o) for o in operations]
    ```

    Hint: Use google.adk.tools.openapi_tool.auth.auth_helpers to construct
    auth_scheme and auth_credential.

    Args:
        name: The name of the tool.
        description: The description of the tool.
        endpoint: Include the base_url, path, and method of the tool.
        operation: Pydantic object or a dict. Representing the OpenAPI Operation
          object
          (https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#operation-object)
        auth_scheme: The auth scheme of the tool. Representing the OpenAPI
          SecurityScheme object
          (https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#security-scheme-object)
        auth_credential: The authentication credential of the tool.
        should_parse_operation: Whether to parse the operation.
    """
    # Gemini restrict the length of function name to be less than 64 characters
    self.name = name[:60]
    self.description = description
    self.endpoint = (
        OperationEndpoint.model_validate_json(endpoint)
        if isinstance(endpoint, str)
        else endpoint
    )
    self.operation = (
        Operation.model_validate_json(operation)
        if isinstance(operation, str)
        else operation
    )
    self.auth_credential, self.auth_scheme = None, None

    self.configure_auth_credential(auth_credential)
    self.configure_auth_scheme(auth_scheme)

    # Private properties
    self.credential_exchanger = AutoAuthCredentialExchanger()
    if should_parse_operation:
      self._operation_parser = OperationParser(self.operation)

  @classmethod
  def from_parsed_operation(cls, parsed: ParsedOperation) -> "RestApiTool":
    """Initializes the RestApiTool from a ParsedOperation object.

    Args:
        parsed: A ParsedOperation object.

    Returns:
        A RestApiTool object.
    """
    operation_parser = OperationParser.load(
        parsed.operation, parsed.parameters, parsed.return_value
    )

    tool_name = to_snake_case(operation_parser.get_function_name())
    generated = cls(
        name=tool_name,
        description=parsed.operation.description
        or parsed.operation.summary
        or "",
        endpoint=parsed.endpoint,
        operation=parsed.operation,
        auth_scheme=parsed.auth_scheme,
        auth_credential=parsed.auth_credential,
    )
    generated._operation_parser = operation_parser
    return generated

  @classmethod
  def from_parsed_operation_str(
      cls, parsed_operation_str: str
  ) -> "RestApiTool":
    """Initializes the RestApiTool from a dict.

    Args:
        parsed: A dict representation of a ParsedOperation object.

    Returns:
        A RestApiTool object.
    """
    operation = ParsedOperation.model_validate_json(parsed_operation_str)
    return RestApiTool.from_parsed_operation(operation)

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Returns the function declaration in the Gemini Schema format."""
    schema_dict = self._operation_parser.get_json_schema()
    parameters = to_gemini_schema(schema_dict)
    function_decl = FunctionDeclaration(
        name=self.name, description=self.description, parameters=parameters
    )
    return function_decl

  def configure_auth_scheme(
      self, auth_scheme: Union[AuthScheme, Dict[str, Any]]
  ):
    """Configures the authentication scheme for the API call.

    Args:
        auth_scheme: AuthScheme|dict -: The authentication scheme. The dict is
          converted to a AuthScheme object.
    """
    if isinstance(auth_scheme, dict):
      auth_scheme = dict_to_auth_scheme(auth_scheme)
    self.auth_scheme = auth_scheme

  def configure_auth_credential(
      self, auth_credential: Optional[Union[AuthCredential, str]] = None
  ):
    """Configures the authentication credential for the API call.

    Args:
        auth_credential: AuthCredential|dict - The authentication credential.
          The dict is converted to an AuthCredential object.
    """
    if isinstance(auth_credential, str):
      auth_credential = AuthCredential.model_validate_json(auth_credential)
    self.auth_credential = auth_credential

  def _prepare_auth_request_params(
      self,
      auth_scheme: AuthScheme,
      auth_credential: AuthCredential,
  ) -> Tuple[List[ApiParameter], Dict[str, Any]]:
    # Handle Authentication
    if not auth_scheme or not auth_credential:
      return

    return credential_to_param(auth_scheme, auth_credential)

  def _prepare_request_params(
      self, parameters: List[ApiParameter], kwargs: Dict[str, Any]
  ) -> Dict[str, Any]:
    """Prepares the request parameters for the API call.

    Args:
        parameters: A list of ApiParameter objects representing the parameters
          for the API call.
        kwargs: The keyword arguments passed to the call function from the Tool
          caller.

    Returns:
        A dictionary containing the  request parameters for the API call. This
        initializes a requests.request() call.

    Example:
        self._prepare_request_params({"input_id": "test-id"})
    """
    method = self.endpoint.method.lower()
    if not method:
      raise ValueError("Operation method not found.")

    path_params: Dict[str, Any] = {}
    query_params: Dict[str, Any] = {}
    header_params: Dict[str, Any] = {}
    cookie_params: Dict[str, Any] = {}

    params_map: Dict[str, ApiParameter] = {p.py_name: p for p in parameters}

    # Fill in path, query, header and cookie parameters to the request
    for param_k, v in kwargs.items():
      param_obj = params_map.get(param_k)
      if not param_obj:
        continue  # If input arg not in the ApiParameter list, ignore it.

      original_k = param_obj.original_name
      param_location = param_obj.param_location

      if param_location == "path":
        path_params[original_k] = v
      elif param_location == "query":
        if v:
          query_params[original_k] = v
      elif param_location == "header":
        header_params[original_k] = v
      elif param_location == "cookie":
        cookie_params[original_k] = v

    # Construct URL
    base_url = self.endpoint.base_url or ""
    base_url = base_url[:-1] if base_url.endswith("/") else base_url
    url = f"{base_url}{self.endpoint.path.format(**path_params)}"

    # Construct body
    body_kwargs: Dict[str, Any] = {}
    request_body = self.operation.requestBody
    if request_body:
      for mime_type, media_type_object in request_body.content.items():
        schema = media_type_object.schema_
        body_data = None

        if schema.type == "object":
          body_data = {}
          for param in parameters:
            if param.param_location == "body" and param.py_name in kwargs:
              body_data[param.original_name] = kwargs[param.py_name]

        elif schema.type == "array":
          for param in parameters:
            if param.param_location == "body" and param.py_name == "array":
              body_data = kwargs.get("array")
              break
        else:  # like string
          for param in parameters:
            # original_name = '' indicating this param applies to the full body.
            if param.param_location == "body" and not param.original_name:
              body_data = (
                  kwargs.get(param.py_name) if param.py_name in kwargs else None
              )
              break

        if mime_type == "application/json" or mime_type.endswith("+json"):
          if body_data is not None:
            body_kwargs["json"] = body_data
        elif mime_type == "application/x-www-form-urlencoded":
          body_kwargs["data"] = body_data
        elif mime_type == "multipart/form-data":
          body_kwargs["files"] = body_data
        elif mime_type == "application/octet-stream":
          body_kwargs["data"] = body_data
        elif mime_type == "text/plain":
          body_kwargs["data"] = body_data

        if mime_type:
          header_params["Content-Type"] = mime_type
        break  # Process only the first mime_type

    filtered_query_params: Dict[str, Any] = {
        k: v for k, v in query_params.items() if v is not None
    }

    request_params: Dict[str, Any] = {
        "method": method,
        "url": url,
        "params": filtered_query_params,
        "headers": header_params,
        "cookies": cookie_params,
        **body_kwargs,
    }

    return request_params

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: Optional[ToolContext]
  ) -> Dict[str, Any]:
    return self.call(args=args, tool_context=tool_context)

  def call(
      self, *, args: dict[str, Any], tool_context: Optional[ToolContext]
  ) -> Dict[str, Any]:
    """Executes the REST API call.

    Args:
        args: Keyword arguments representing the operation parameters.
        tool_context: The tool context (not used here, but required by the
          interface).

    Returns:
        The API response as a dictionary.
    """
    # Prepare auth credentials for the API call
    tool_auth_handler = ToolAuthHandler.from_tool_context(
        tool_context, self.auth_scheme, self.auth_credential
    )
    auth_result = tool_auth_handler.prepare_auth_credentials()
    auth_state, auth_scheme, auth_credential = (
        auth_result.state,
        auth_result.auth_scheme,
        auth_result.auth_credential,
    )

    if auth_state == "pending":
      return {
          "pending": True,
          "message": "Needs your authorization to access your data.",
      }

    # Attach parameters from auth into main parameters list
    api_params, api_args = self._operation_parser.get_parameters().copy(), args
    if auth_credential:
      # Attach parameters from auth into main parameters list
      auth_param, auth_args = self._prepare_auth_request_params(
          auth_scheme, auth_credential
      )
      if auth_param and auth_args:
        api_params = [auth_param] + api_params
        api_args.update(auth_args)

    # Got all parameters. Call the API.
    request_params = self._prepare_request_params(api_params, api_args)
    response = requests.request(**request_params)

    # Parse API response
    try:
      response.raise_for_status()  # Raise HTTPError for bad responses
      return response.json()  # Try to decode JSON
    except requests.exceptions.HTTPError:
      error_details = response.content.decode("utf-8")
      return {
          "error": (
              f"Tool {self.name} execution failed. Analyze this execution error"
              " and your inputs. Retry with adjustments if applicable. But"
              " make sure don't retry more than 3 times. Execution Error:"
              f" {error_details}"
          )
      }
    except ValueError:
      return {"text": response.text}  # Return text if not JSON

  def __str__(self):
    return (
        f'RestApiTool(name="{self.name}", description="{self.description}",'
        f' endpoint="{self.endpoint}")'
    )

  def __repr__(self):
    return (
        f'RestApiTool(name="{self.name}", description="{self.description}",'
        f' endpoint="{self.endpoint}", operation="{self.operation}",'
        f' auth_scheme="{self.auth_scheme}",'
        f' auth_credential="{self.auth_credential}")'
    )



================================================
FILE: src/google/adk/tools/openapi_tool/openapi_spec_parser/tool_auth_handler.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from typing import Literal
from typing import Optional

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from ....auth.auth_credential import AuthCredential
from ....auth.auth_credential import AuthCredentialTypes
from ....auth.auth_schemes import AuthScheme
from ....auth.auth_schemes import AuthSchemeType
from ....auth.auth_tool import AuthConfig
from ...tool_context import ToolContext
from ..auth.credential_exchangers.auto_auth_credential_exchanger import AutoAuthCredentialExchanger
from ..auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
from ..auth.credential_exchangers.base_credential_exchanger import BaseAuthCredentialExchanger

logger = logging.getLogger(__name__)

AuthPreparationState = Literal["pending", "done"]


class AuthPreparationResult(BaseModel):
  """Result of the credential preparation process."""

  state: AuthPreparationState
  auth_scheme: Optional[AuthScheme] = None
  auth_credential: Optional[AuthCredential] = None


class ToolContextCredentialStore:
  """Handles storage and retrieval of credentials within a ToolContext."""

  def __init__(self, tool_context: ToolContext):
    self.tool_context = tool_context

  def get_credential_key(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> str:
    """Generates a unique key for the given auth scheme and credential."""
    scheme_name = (
        f"{auth_scheme.type_.name}_{hash(auth_scheme.model_dump_json())}"
        if auth_scheme
        else ""
    )
    credential_name = (
        f"{auth_credential.auth_type.value}_{hash(auth_credential.model_dump_json())}"
        if auth_credential
        else ""
    )
    # no need to prepend temp: namespace, session state is a copy, changes to
    # it won't be persisted , only changes in event_action.state_delta will be
    # persisted. temp: namespace will be cleared after current run. but tool
    # want access token to be there stored across runs

    return f"{scheme_name}_{credential_name}_existing_exchanged_credential"

  def get_credential(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> Optional[AuthCredential]:
    if not self.tool_context:
      return None

    token_key = self.get_credential_key(auth_scheme, auth_credential)
    # TODO try not to use session state, this looks a hacky way, depend on
    # session implementation, we don't want session to persist the token,
    # meanwhile we want the token shared across runs.
    serialized_credential = self.tool_context.state.get(token_key)
    if not serialized_credential:
      return None
    return AuthCredential.model_validate(serialized_credential)

  def store_credential(
      self,
      key: str,
      auth_credential: Optional[AuthCredential],
  ):
    if self.tool_context:
      serializable_credential = jsonable_encoder(
          auth_credential, exclude_none=True
      )
      self.tool_context.state[key] = serializable_credential

  def remove_credential(self, key: str):
    del self.tool_context.state[key]


class ToolAuthHandler:
  """Handles the preparation and exchange of authentication credentials for tools."""

  def __init__(
      self,
      tool_context: ToolContext,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
      credential_exchanger: Optional[BaseAuthCredentialExchanger] = None,
      credential_store: Optional["ToolContextCredentialStore"] = None,
  ):
    self.tool_context = tool_context
    self.auth_scheme = (
        auth_scheme.model_copy(deep=True) if auth_scheme else None
    )
    self.auth_credential = (
        auth_credential.model_copy(deep=True) if auth_credential else None
    )
    self.credential_exchanger = (
        credential_exchanger or AutoAuthCredentialExchanger()
    )
    self.credential_store = credential_store
    self.should_store_credential = True

  @classmethod
  def from_tool_context(
      cls,
      tool_context: ToolContext,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
      credential_exchanger: Optional[BaseAuthCredentialExchanger] = None,
  ) -> "ToolAuthHandler":
    """Creates a ToolAuthHandler instance from a ToolContext."""
    credential_store = ToolContextCredentialStore(tool_context)
    return cls(
        tool_context,
        auth_scheme,
        auth_credential,
        credential_exchanger,
        credential_store,
    )

  def _handle_existing_credential(
      self,
  ) -> Optional[AuthPreparationResult]:
    """Checks for and returns an existing, exchanged credential."""
    if self.credential_store:
      existing_credential = self.credential_store.get_credential(
          self.auth_scheme, self.auth_credential
      )
      if existing_credential:
        return AuthPreparationResult(
            state="done",
            auth_scheme=self.auth_scheme,
            auth_credential=existing_credential,
        )
    return None

  def _exchange_credential(
      self, auth_credential: AuthCredential
  ) -> Optional[AuthPreparationResult]:
    """Handles an OpenID Connect authorization response."""

    exchanged_credential = None
    try:
      exchanged_credential = self.credential_exchanger.exchange_credential(
          self.auth_scheme, auth_credential
      )
    except Exception as e:
      logger.error("Failed to exchange credential: %s", e)
    return exchanged_credential

  def _store_credential(self, auth_credential: AuthCredential) -> None:
    """stores the auth_credential."""

    if self.credential_store:
      key = self.credential_store.get_credential_key(
          self.auth_scheme, self.auth_credential
      )
      self.credential_store.store_credential(key, auth_credential)

  def _reqeust_credential(self) -> None:
    """Handles the case where an OpenID Connect or OAuth2 authentication request is needed."""
    if self.auth_scheme.type_ in (
        AuthSchemeType.openIdConnect,
        AuthSchemeType.oauth2,
    ):
      if not self.auth_credential or not self.auth_credential.oauth2:
        raise ValueError(
            f"auth_credential is empty for scheme {self.auth_scheme.type_}."
            "Please create AuthCredential using OAuth2Auth."
        )

      if not self.auth_credential.oauth2.client_id:
        raise AuthCredentialMissingError(
            "OAuth2 credentials client_id is missing."
        )

      if not self.auth_credential.oauth2.client_secret:
        raise AuthCredentialMissingError(
            "OAuth2 credentials client_secret is missing."
        )

    self.tool_context.request_credential(
        AuthConfig(
            auth_scheme=self.auth_scheme,
            raw_auth_credential=self.auth_credential,
        )
    )
    return None

  def _get_auth_response(self) -> AuthCredential:
    return self.tool_context.get_auth_response(
        AuthConfig(
            auth_scheme=self.auth_scheme,
            raw_auth_credential=self.auth_credential,
        )
    )

  def _request_credential(self, auth_config: AuthConfig):
    if not self.tool_context:
      return
    self.tool_context.request_credential(auth_config)

  def prepare_auth_credentials(
      self,
  ) -> AuthPreparationResult:
    """Prepares authentication credentials, handling exchange and user interaction."""

    # no auth is needed
    if not self.auth_scheme:
      return AuthPreparationResult(state="done")

    # Check for existing credential.
    existing_result = self._handle_existing_credential()
    if existing_result:
      return existing_result

    # fetch credential from adk framework
    # Some auth scheme like OAuth2 AuthCode & OpenIDConnect may require
    # multi-step exchange:
    # client_id , client_secret -> auth_uri -> auth_code -> access_token
    # -> bearer token
    # adk framework supports exchange access_token already
    fetched_credential = self._get_auth_response() or self.auth_credential

    exchanged_credential = self._exchange_credential(fetched_credential)

    if exchanged_credential:
      self._store_credential(exchanged_credential)
      return AuthPreparationResult(
          state="done",
          auth_scheme=self.auth_scheme,
          auth_credential=exchanged_credential,
      )
    else:
      self._reqeust_credential()
      return AuthPreparationResult(
          state="pending",
          auth_scheme=self.auth_scheme,
          auth_credential=self.auth_credential,
      )



================================================
FILE: src/google/adk/tools/retrieval/__init__.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base_retrieval_tool import BaseRetrievalTool
from .files_retrieval import FilesRetrieval
from .llama_index_retrieval import LlamaIndexRetrieval

__all__ = [
    'BaseRetrievalTool',
    'FilesRetrieval',
    'LlamaIndexRetrieval',
]

try:
  from .vertex_ai_rag_retrieval import VertexAiRagRetrieval

  __all__.append('VertexAiRagRetrieval')
except ImportError:
  import logging

  logger = logging.getLogger(__name__)
  logger.debug(
      'The Vertex sdk is not installed. If you want to use the Vertex RAG with'
      ' agents, please install it. If not, you can ignore this warning.'
  )



================================================
FILE: src/google/adk/tools/retrieval/base_retrieval_tool.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.genai import types
from typing_extensions import override

from ..base_tool import BaseTool


class BaseRetrievalTool(BaseTool):

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                'query': types.Schema(
                    type=types.Type.STRING,
                    description='The query to retrieve.',
                ),
            },
        ),
    )



================================================
FILE: src/google/adk/tools/retrieval/files_retrieval.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides data for the agent."""

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

from .llama_index_retrieval import LlamaIndexRetrieval


class FilesRetrieval(LlamaIndexRetrieval):

  def __init__(self, *, name: str, description: str, input_dir: str):

    self.input_dir = input_dir

    print(f'Loading data from {input_dir}')
    retriever = VectorStoreIndex.from_documents(
        SimpleDirectoryReader(input_dir).load_data()
    ).as_retriever()
    super().__init__(name=name, description=description, retriever=retriever)



================================================
FILE: src/google/adk/tools/retrieval/llama_index_retrieval.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides data for the agent."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from typing_extensions import override

from ..tool_context import ToolContext
from .base_retrieval_tool import BaseRetrievalTool

if TYPE_CHECKING:
  from llama_index.core.base.base_retriever import BaseRetriever


class LlamaIndexRetrieval(BaseRetrievalTool):

  def __init__(self, *, name: str, description: str, retriever: BaseRetriever):
    super().__init__(name=name, description=description)
    self.retriever = retriever

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    return self.retriever.retrieve(args['query'])[0].text



================================================
FILE: src/google/adk/tools/retrieval/vertex_ai_rag_retrieval.py
================================================
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A retrieval tool that uses Vertex AI RAG to retrieve data."""

from __future__ import annotations

import logging
from typing import Any
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override
from vertexai.preview import rag

from ..tool_context import ToolContext
from .base_retrieval_tool import BaseRetrievalTool

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest

logger = logging.getLogger(__name__)


class VertexAiRagRetrieval(BaseRetrievalTool):
  """A retrieval tool that uses Vertex AI RAG (Retrieval-Augmented Generation) to retrieve data."""

  def __init__(
      self,
      *,
      name: str,
      description: str,
      rag_corpora: list[str] = None,
      rag_resources: list[rag.RagResource] = None,
      similarity_top_k: int = None,
      vector_distance_threshold: float = None,
  ):
    super().__init__(name=name, description=description)
    self.vertex_rag_store = types.VertexRagStore(
        rag_corpora=rag_corpora,
        rag_resources=rag_resources,
        similarity_top_k=similarity_top_k,
        vector_distance_threshold=vector_distance_threshold,
    )

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    # Use Gemini built-in Vertex AI RAG tool for Gemini 2 models.
    if llm_request.model and llm_request.model.startswith('gemini-2'):
      llm_request.config = (
          types.GenerateContentConfig()
          if not llm_request.config
          else llm_request.config
      )
      llm_request.config.tools = (
          [] if not llm_request.config.tools else llm_request.config.tools
      )
      llm_request.config.tools.append(
          types.Tool(
              retrieval=types.Retrieval(vertex_rag_store=self.vertex_rag_store)
          )
      )
    else:
      # Add the function declaration to the tools
      await super().process_llm_request(
          tool_context=tool_context, llm_request=llm_request
      )

  @override
  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:

    response = rag.retrieval_query(
        text=args['query'],
        rag_resources=self.vertex_rag_store.rag_resources,
        rag_corpora=self.vertex_rag_store.rag_corpora,
        similarity_top_k=self.vertex_rag_store.similarity_top_k,
        vector_distance_threshold=self.vertex_rag_store.vector_distance_threshold,
    )

    logging.debug('RAG raw response: %s', response)

    return (
        f'No matching result found with the config: {self.vertex_rag_store}'
        if not response.contexts.contexts
        else [context.text for context in response.contexts.contexts]
    )


