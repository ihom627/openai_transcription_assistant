This is an OpenAI based assistant that uses the Whisper service to transcribe an input audio file to an output a text summary file

Divided this into two parts:
  first part processes an mp3 audio file and outputs a text file transcription.txt 
  second part processes the text file transcription.txt and outputs a summary file ./summary.txt


NOTES: running the OpenAI whisper mp3 audio to text service is expensive, so only run when necessary.

originally from https://github.com/GEScott71/GPT_Meeting_Minutes/tree/master


Step1) env setup

brew install ffmpeg

Step2) set OPENAI_API_KEY

export OPENAI_API_KEY="XXX"

Step3) create temp Data directory

mkdir Data

Step3) python3 assistant_generate_transcript.py Ford_Q3_2023_earnings_call.mp3_segment_1.mp3

output in transcript.txt

Step4) python3 assistant_generate_summary.py

output in summary.txt 

 
