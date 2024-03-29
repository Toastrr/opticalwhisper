from faster_whisper import WhisperModel
from math import floor
from time import monotonic
from argparse import ArgumentParser
from os.path import dirname
from concurrent.futures import ThreadPoolExecutor

#import argparse
#import logging

LANGUAGE_CODES = ("af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de",
                  "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht",
                  "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt",
                  "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl",
                  "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta",
                  "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh",)
MODELS = ("large-v2", "large", "medium", "medium.en", "small", "small.en", "base", "base.en", "tiny", "tiny.en")


def convert_timestamp(seconds: float) -> str:
    remaining_seconds = seconds
    hours = floor(remaining_seconds / 3600)
    remaining_seconds = remaining_seconds - (hours * 3600)
    minutes = floor(remaining_seconds / 60)
    remaining_seconds = remaining_seconds - (minutes * 60)
    return (f"{hours:02d}:{minutes:02d}:{floor(remaining_seconds):02d},"
            f"{f'{(remaining_seconds - floor(remaining_seconds)):.3f}'[2:]}")


def convert_seconds_to_hms(seconds: float) -> str:
    remaining_seconds = seconds
    hours = floor(remaining_seconds / 3600)
    remaining_seconds = remaining_seconds - (hours * 3600)
    minutes = floor(remaining_seconds / 60)
    remaining_seconds = remaining_seconds - (minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{floor(remaining_seconds):02d}"


def write_subtitles(whisper_results, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write("1\n00:00:00,000 --> 00:00:03,000\n[Optical Whisper generated subtitles]\n\n")
        for i in whisper_results:
            start = convert_timestamp(i.get('start'))
            end = convert_timestamp(i.get('end'))
            file.write(f"{i.get('id') + 1}\n{start} --> {end}\n{i.get('text').lstrip()}\n\n")
        last = whisper_results[-1]
        file.write(f"{last.get('id')+1}\n"
                   f"{convert_timestamp(last.get('start'))} --> {convert_timestamp(last.get('end'))}\n"
                   f"[Optical Whisper generated subtitles]\n")


def chunk_write_subtitles(result, filename):
    with open(filename, "a", encoding="utf-8") as file:
        start = convert_timestamp(result.get('start'))
        end = convert_timestamp(result.get('end'))
        file.write(f"{result.get('id') + 1}\n{start} --> {end}\n{result.get('text').lstrip()}\n\n")


def transcribe(model, file: str, beam_size: int = 5, task: str = "transcribe", language=None, vad_filter: bool = True):
    segments, info = model.transcribe(file, beam_size=beam_size, task=task, language=language, vad_filter=vad_filter)
    return model, info._asdict(), segments


def process_file(model, file, args):
    # logging.debug(f"Analysing {file}")
    print(f"\n[{file}]: Analysing...")
    model, info, segments = transcribe(model, file, beam_size=args.beam_size, task=args.task,
                                       vad_filter=args.vad_filter, language=args.language)
    # logging.debug(f"File analysed with info {info}")
    print(f"[{file}]: Detected language {info.get('language')} with {info.get('language_probability')} probability")
    print(f"[{file}]: Transcribing Duration of {info.get('duration_after_vad')} seconds")
    print(f"[{file}]: Beginning transcription")
    # logging.info("Starting transcription")
    start = monotonic()
    result = []
    for i in segments:
        seg = i._asdict()
        result.append(seg)
        elapsed_seconds = (monotonic() - start)
        percent_completed = round((float(seg.get('end')) / float(info.get('duration')) * 100), 2)
        estimated_remaining_seconds = ((100 / percent_completed) - 1) * elapsed_seconds
        print(f"[{file}]: Elapsed time: {convert_seconds_to_hms(elapsed_seconds)} "
              f"Completed: {percent_completed}% {round(seg.get('end'), 2)}/{round(info.get('duration'), 2)} seconds "
              f"Estimated {convert_seconds_to_hms(estimated_remaining_seconds)} remaining"
              , end="\r")

    end = monotonic()
    # logging.info("Finished transcription")
    print(f"[{file}]: Finished transcription in {round(end - start, 3)} seconds")
    print(f"[{file}]: Writing subtitles to {file}.srt")
    write_subtitles(result, f"{file}.srt")
    print(f"[{file}]: Completed")
    return model


def main():
    parser = ArgumentParser(prog="Optical Whisper", description="Transcribes/Translate audio")
    parser.add_argument("files", type=str, nargs="+",
                        help="Files to process")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size")
    parser.add_argument("--compute-type", type=str,
                        choices=("float16", "int8_float16", "int8", "float32", "bfloat16"),
                        default="float16",
                        help="Compute type")
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="CPU threads for cpu compute")
    parser.add_argument("--device", type=str, choices=("cuda", "cpu"), default="cuda",
                        help="Device to run on")
    parser.add_argument("--device-index", type=str, default="0",
                        help="comma delimited list of cuda devices index")
    parser.add_argument("--language", type=str, default=None,
                        help="Specify audio language")
    parser.add_argument("--model", type=str, choices=MODELS, default="large-v2",
                        help="Whisper model")
    parser.add_argument("--task", type=str, choices=("transcribe", "translate"), default="transcribe",
                        help="Transcribe or Translate")
    parser.add_argument("--vad-filter", type=bool, choices=(True, False), default=True,
                        help="Enable or disable the vad filter")
    args = parser.parse_args()

    whisper_model = f"{dirname(__file__)}/models/{args.compute_type}/{args.model}"
    args.device_index = [int(i) for i in str(args.device_index).split(",")]
    #logging.info(f"Initialising with the following args {args}")
    print("Loading Model...")
    #logging.debug("Loading model")
    model = WhisperModel(whisper_model, device=args.device, compute_type=args.compute_type,
                         device_index=args.device_index, local_files_only=True, cpu_threads=args.cpu_threads)
    #logging.debug("Model Loaded")
    print("Loaded Model")
    print("Begin Processing Files...")

    if len(args.files) <= 1 or (len(args.device_index) <= 1 or args.cpu_threads <= 1):
        for file in args.files:
            process_file(model, file, args)
    elif len(args.device_index) > 1:
        with ThreadPoolExecutor(max_workers=len(args.device_index)) as thread_executor:
            for file in args.files:
                thread_executor.submit(process_file, model, file, args)
    elif args.cpu_threads > 1:
        with ThreadPoolExecutor(max_workers=args.cpu_threads) as thread_executor:
            for file in args.files:
                thread_executor.submit(process_file, model, file, args)

    print("\nFinished Processing all Files, Exiting...")
    del model


if __name__ == '__main__':
    main()
