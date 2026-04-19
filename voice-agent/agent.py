import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent,AgentServer,AgentSession,JobContext,room_io,TurnHandlingOptions
from livekit.plugins import silero
from livekit.plugins.noise_cancellation import BVC
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import stt, llm, tts,inference
from livekit.agents import AgentStateChangeEvent, metrics,MetricsCollectedEvent
import time
logger=logging.getLogger(__name__)
load_dotenv()

class Assistant(Agent):
    def __init__(self)->None:
        super().__init__(instructions="You are an upbeat,slightly sarcastic voice AI for tech support." 
                         "Help the caller fix issues without rambling, and keep replies under 3 sentences.")

server = AgentServer()
@server.rtc_session()  #rtc-real time communiccation session 
async def entrypoint(CTX: JobContext):
    assistant = Assistant() # created class to inherit the agent class without writing evrytime the instruction
    session = AgentSession(
         stt=stt.FallbackAdapter([
             inference.STT.from_model_string("assemblyai/universal-streaming:en"),
             inference.STT.from_model_string("deepgram/nova-3"),
         ]),
        # stt="assemblyai/universal-streaming:en",  # assemblyai is speech to text model or u can use deepagram which is alos free
        llm=llm.FallbackAdapter([
            inference.LLM(model="openai/gpt-4.1-mini"),
            inference.LLM(model="google/gemini-2.5-flash"),
        ]),
        # llm="openai/gpt-4.1-mini",   # llm model to genrate response
        tts=tts.FallbackAdapter([
            inference.TTS.from_model_string("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
            inference.TTS.from_model_string("inworld/inworld-tts-1"),
        ]),
        # tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",  # text to speech model to genrate audio response
        vad=silero.VAD.load(),  # vad- voice activity detection model to detect when user is speaking and when not
        # turn_detector=MultilingualModel() # to detect the language of the user and respond in the same language (multilingual model)
    #     turn_handling=TurnHandlingOptions(
    #     turn_detection=MultilingualModel(),
    # ),
       preemptive_generation=True, # to start generating response as soon as the user stops speaking 
  )
    
    usage_collector=metrics.UsageCollector() # to collect the usage of the models and the agent
    last_eou_metrics:metrics.EOUMetrics|None=None # to store the last end of utterance metrics
    @session.on("metrics_collected")
    async def on_metrics_collected(ev:MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics=ev.metrics
        
        metrics.log_metrics(ev.metrics) # to log the metrics in the console
        usage_collector.collect(ev.metrics) # to collect the metrics in the usage collector

    async def log_usage():
        summary=usage_collector.summary() # to get the summary of the usage
        logger.info(f"Usage summary: {summary}") # to log the usage summary 

    CTX.add_shutdown_callback(log_usage) # to log the usage summary when the session is shutdown

    @session.on("agent_state_change")
    async def on_agent_state_change(ev: AgentStateChangeEvent):
        if ev.new_state =="speaking":
            if last_eou_metrics:
                elapsed=time.time()-last_eou_metrics.timestamp
                logger.info(f"Time taken from end of user utterance to start of agent response: {elapsed:.2f} seconds")
        

    await session.start(
        assistant,
        room=CTX.room, # room is the livekit room where the context will take place and the agent will interact with the user
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=BVC(),
            )
        ) # room_io is the audio and video input and output manage options and noise cancellation also added
    )

if __name__ == "__main__":   # this is the main function to run the server and start the agent
    logging.basicConfig(level=logging.INFO) # set the logging level to info to see the logs in the console
    agents.cli.run_app(server) # to run server and start agent