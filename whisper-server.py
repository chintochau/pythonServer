from flask import Flask, request
from faster_whisper import WhisperModel
import time
import os
import uuid
from progress.bar import Bar
from moviepy.editor import  AudioFileClip
from dotenv import load_dotenv
load_dotenv()
import cloudinary
import json
import cloudinary.uploader
config = cloudinary.config(secure=True)

app = Flask(__name__)

model_size = "large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16",download_root=".")

transcription_progress = {}

def format_srt_time(start_time):
    seconds = int(start_time % 60)
    total_minutes = int(start_time // 60)
    minutes = total_minutes % 60
    hours = int(total_minutes // 60)
    milliseconds = int((start_time - int(start_time)) * 1000)
    formatted_time = "%02d:%02d:%02d,%03d" % (hours, minutes, seconds, milliseconds)
    return formatted_time

# delete file on cloudinary
def delete_file_on_cloudinary(public_id, resource_type):
    cloudinary.uploader.destroy(public_id, resource_type = resource_type)
    pass


@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['file']
    file_extension = os.path.splitext(file.filename)[-1]
    unique_filename = str(uuid.uuid4()) + file_extension  # Generate a unique filename
    file_path = os.path.join("uploads/", unique_filename)
    file.save(file_path)

    segments, info = model.transcribe(file_path, beam_size=5, chunk_length=30,vad_filter=True)

    clip = AudioFileClip(file_path)
    duration = clip.duration
    clip.close()
    bar = Bar('Processing', max=duration, suffix='%(percent)d%% %(elapsed)ds')


    output = ""
    for segment in segments:
        output += "%s\n" % (segment.id)
        output += "%s --> %s\n" % (format_srt_time(segment.start), format_srt_time(segment.end))
        output += "%s\n\n" % (segment.text)
        bar.goto(segment.end)
    bar.goto(duration)
    bar.finish()
    print("video duration: ", duration)
    
    os.remove(file_path)

    return output

@app.route('/transcribe_with_link', methods=['POST'])
def transcribe_with_link():
    link = request.form['link']
    public_id = request.form['publicId']
    resource_type = request.form['resourceType']
    transcriptionId = request.form['transcriptionId']
    
    # Initialize progress
    transcription_progress[transcriptionId] = 0

    segments, info = model.transcribe(link, beam_size=5, chunk_length=30,vad_filter=True)

    clip = AudioFileClip(link)
    duration = clip.duration
    clip.close()
    bar = Bar('Processing', max=duration, suffix='%(percent)d%% %(elapsed)ds')

    output = ""
    for segment in segments:
        output += "%s\n" % (segment.id)
        output += "%s --> %s\n" % (format_srt_time(segment.start), format_srt_time(segment.end))
        output += "%s\n\n" % (segment.text)

        # Update progress
        transcription_progress[transcriptionId] = segment.end / duration * 100
        bar.goto(segment.end)
    bar.goto(duration)
    transcription_progress[transcriptionId] = 100
    bar.finish()
    delete_file_on_cloudinary(public_id, resource_type)

    # delete the file on cloudinary after processing
    # to be implemented

    return output

@app.route('/check_transcription_progress', methods=['GET'])
def check_transcription_progress():
    transcriptionId = request.args.get('transcriptionId')
    # if not progress, return 0
    progress = transcription_progress[transcriptionId] if transcriptionId in transcription_progress else 0
    print("progress: ", progress)
    return str(progress)


@app.route('/', methods=['GET'])
def check_transcription_status():
    # Implement the logic to check the status of the transcription process
    # Example: return a message indicating that the service is running
    return "Transcription service is running"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
