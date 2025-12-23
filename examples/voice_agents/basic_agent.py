import logging


from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Interruption configuration

IGNORE_WORDS = {"yeah", "ok", "okay", "hmm", "uh", "uh-huh", "right", "aha"}
INTERRUPT_WORDS = {"stop", "wait", "pause", "no"}
INTERRUPT_PHRASES = {"hold on"}

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly. You interact with users via voice. "
                "Keep responses concise and to the point. "
                "Do not use emojis, markdown, or special characters. "
                "You are curious, friendly, and speak English."
            ),
        )
        self.is_speaking = False

    async def on_enter(self):
        self.session.generate_reply()

    # --- Track agent speaking state ---
    async def on_agent_speech_start(self):
        self.is_speaking = True

    async def on_agent_speech_end(self):
        self.is_speaking = False

    # --- Core interruption logic ---
    async def on_user_speech(self, transcript: str):
        text = transcript.lower().strip()

        logger.info(
            f"[INTERRUPT] speaking={self.is_speaking} text='{text}'"
        )
        # Normalize tokens safely (handles punctuation)
        tokens = [t.strip(".,!?") for t in text.split()]

        # Agent is speaking
        if self.is_speaking:
            # Hard interrupt (semantic command)
            if (
                any(phrase in text for phrase in INTERRUPT_PHRASES)
                or any(word in tokens for word in INTERRUPT_WORDS)
            ):
                await self.session.stop_speaking()
                await self.session.handle_user_input(transcript)
                return

            # Backchannel only → IGNORE COMPLETELY
            if tokens and all(t in IGNORE_WORDS for t in tokens):
                return

            # Mixed or meaningful speech → interrupt
            await self.session.stop_speaking()
            await self.session.handle_user_input(transcript)
            return

        # Agent is silent → normal behavior
        await self.session.handle_user_input(transcript)

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."



server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt="deepgram/nova-3",
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm="openai/gpt-4.1-mini",
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # uncomment to enable the Krisp BVC noise cancellation
                # noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
