import argparse
import hashlib
import os
import traceback
import wave
import matplotlib.pyplot as plt
import numba.cuda.cudadrv.ndarray
import numpy as np
from numba import cuda
import pickle

CACHE_LOC = ".auto_correlation_audio_extraction_cache"


def wave_to_np(wav: wave.Wave_read):
    wav.rewind()
    frames: bytes = wav.readframes(-1)
    return np.array(
        [
            int.from_bytes(
                frames[i:i + wav.getsampwidth()],
                "little",
                signed=True
            ) for i in range(0, len(frames) - 1, wav.getsampwidth())
        ]
    )


@numba.jit(nopython=True, parallel=True, nogil=True)
def calculate_correlation_data_cpu(audio: np.ndarray,
                                   sample_start: int = 0,
                                   sample_len: int = 200_000):
    correlation_data = np.zeros(len(audio), dtype=np.int64)
    sample = audio[sample_start: sample_start + sample_len]

    for pos in range(len(audio)):
        if pos % 2000 == 0:
            with numba.objmode():
                print(f"\r* "
                      f"{((pos + 1) / len(audio) * 100).__format__('.2f')}% "
                      f"{(pos + 1).__format__('_')}/"
                      f"{len(audio).__format__('_')}", end="")

        audio_sample = audio[pos:min(pos + sample_len, len(audio))]
        _sample = sample[sample_start: sample_start + min(sample_len, len(audio_sample))]

        # not using np.average because it's unsupported by numba, instead using np.sum / len
        diff = np.absolute(_sample - audio_sample)
        correlation_data[pos] = np.sum(diff) // len(diff)

    print()

    return correlation_data


@cuda.jit
def _calculate_correlation_gpu(sample_data: numba.cuda.devicearray.DeviceNDArray,
                               wav_data: numba.cuda.devicearray.DeviceNDArray,
                               output: numba.cuda.devicearray.DeviceNDArray):
    # get position in gpu grid
    # noinspection PyArgumentList
    # noinspection PyTypeChecker
    start_position: int = cuda.grid(1)

    diffsum = 0
    iterations = min(len(sample_data), len(wav_data) - start_position)
    for i in range(iterations):
        byte_1 = wav_data[start_position + i]
        byte_2 = sample_data[i]

        diffsum += abs(byte_1 - byte_2)

    output[start_position] = diffsum / iterations


def calculate_correlation_data_gpu(audio: np.ndarray,
                                   sample_start: int = 0,
                                   sample_len: int = 200_000, ):
    threads_per_block = 32

    # max_arr_len = 1024 * 32
    print("* Uploading to GPU and preparing media...")

    sample_data_gpu = cuda.to_device(audio[sample_start:sample_start + sample_len])

    wav_data_gpu = cuda.to_device(audio)

    output = np.zeros(len(audio), dtype=np.int64)

    blocks_per_grid = audio.size // threads_per_block
    print(f"> Starting with {threads_per_block=:_} {blocks_per_grid=:_}")
    print(f"> {len(audio) =:_}")
    _calculate_correlation_gpu[blocks_per_grid, threads_per_block](sample_data_gpu, wav_data_gpu, output)

    return output


def get_correlation_data(audio: np.ndarray,
                         dont_cache: bool = False,
                         force_recalculate: bool = False,
                         force_cpu: bool = False,
                         correlation_data_interval: tuple[int, int] = (20, -200),
                         sample_start: int = 0,
                         sample_len: int = 200_000,
                         ):
    def get_fn_from_hash(hash_str: str):
        return f"enhanced_correlation_data_{hash_str}.pickle"

    if not force_recalculate or not dont_cache:
        print("* Calculating audio hash")
        audio_hash = hashlib.sha256(audio).hexdigest()

    if not force_recalculate:
        print("* Searching cache")
        try:
            fn = get_fn_from_hash(audio_hash)
            if fn in os.listdir(CACHE_LOC):
                with open(os.path.join(CACHE_LOC, fn), "rb") as f:
                    print("> Found something")
                    return pickle.load(f)
            else:
                raise FileNotFoundError()
        except FileNotFoundError:
            print("> Found nothing")

    if force_cpu:
        print("* Forced cpu correlation calculation")
        correlation_data = calculate_correlation_data_cpu(
            audio,
            sample_start=sample_start, sample_len=sample_len
        )
    else:
        try:
            print("* Using gpu correlation calculation")
            correlation_data = calculate_correlation_data_gpu(
                audio,
                sample_start=sample_start, sample_len=sample_len
            )
        except Exception as e:
            print(f"> Couldn't use gpu correlation calculation: {e!r}")
            traceback.print_exc()

            print(f"> Falling back to cpu correlation data calculation method")
            correlation_data = calculate_correlation_data_cpu(
                audio,
                sample_start=sample_start, sample_len=sample_len
            )

    print("* Enhancing correlation data")
    correlation_data = correlation_data[correlation_data_interval[0]:correlation_data_interval[1]]
    correlation_data = np.diff(correlation_data)
    correlation_data **= 2

    if not dont_cache:
        print("* Storing to cache")
        os.makedirs(
            CACHE_LOC,
            exist_ok=True
        )
        filename = os.path.join(CACHE_LOC, get_fn_from_hash(audio_hash))
        with open(filename, "wb") as f:
            pickle.dump(correlation_data, f)

    return correlation_data


def extract_time_codes(correlation_data: np.array,
                       correlation_threshold=500,
                       max_cluster_len=100_000):
    clusters: list[list[int]] = [[]]
    last_cluster = 0

    for i, sample in enumerate(correlation_data):
        if sample > correlation_threshold:
            if last_cluster > max_cluster_len:
                clusters.append([])

            last_cluster = 0
            clusters[-1].append(i)

        last_cluster += 1

    time_codes: list[int] = []
    for cluster in clusters:
        if not cluster:
            continue
        time_codes.append(max(cluster))

    return time_codes


def sum_audio(audio: np.ndarray, time_codes: list[int]) -> np.ndarray:
    out = np.zeros(len(audio), dtype=int)

    for i, time_code in enumerate(time_codes):
        print(f"\r* {(i + 1) / len(time_codes) * 100:.2f}% {i + 1:_}/{len(time_codes):_}", end="")

        # print("cutting")
        # cutt_audio = cut_audio(audio[time_code:])

        # print("adding")
        out[:len(audio) - time_code] += audio[time_code:]

    print()

    return out // len(time_codes)


def get_correlation_threshold(correlation_data: np.ndarray) -> float:
    return np.partition(correlation_data, -3)[-3]/5


def extract_audio(audio: np.ndarray, *,
                  force_recalculate: bool = False,
                  dont_cache: bool = False,
                  force_cpu: bool = False,
                  plot_correlation_data: bool = False,
                  correlation_data_interval: tuple[int, int] = (20, -200),
                  sample_start: int = 0,
                  sample_len: int = 200_000,
                  correlation_threshold,
                  max_cluster_len):
    print("* Getting correlation data")

    correlation_data = get_correlation_data(
        audio,
        dont_cache=dont_cache,
        force_recalculate=force_recalculate,
        force_cpu=force_cpu,
        correlation_data_interval=correlation_data_interval,
        sample_start=sample_start,
        sample_len=sample_len,
    )

    if plot_correlation_data:
        print("* Plotting correlation data")
        plt.plot(correlation_data)
        plt.show()

    if not correlation_threshold:
        print("* Calculating correlation threshold")
        correlation_threshold = get_correlation_threshold(correlation_data)

        print(f"> {correlation_threshold:.2f}")

    print("* Extracting time codes (repeats)")
    time_codes = extract_time_codes(
        correlation_data,
        correlation_threshold=correlation_threshold,
        max_cluster_len=max_cluster_len
    )

    print(f"> Found {len(time_codes)} occurrences")

    print("* Summing to final output")
    summed_audio = sum_audio(
        audio,
        time_codes=time_codes
    )

    return summed_audio


def save(audio: np.ndarray, wav: wave.Wave_write):
    wav.setnchannels(1)

    audio = np.array(audio, dtype=f"<i{wav.getsampwidth()}")

    wav.writeframes(audio.tobytes())

    wav.close()


def main():
    parser = argparse.ArgumentParser(description="Extract repeating noisy audio with the help of autocorrelation")
    parser.add_argument("-input-file", "-i", type=str, help="The input file (wav).", required=True)
    parser.add_argument("-output-file", "-o", type=str, default="out.wav",
                        help="The output file (wav). Default is out.wav")
    parser.add_argument("-sample-start", "-ss",
                        type=int, default=0,
                        help="Specifies the start position (in nr. of samples) of the sample "
                             "with which the full data is auto correlated with. Default is 0.")
    parser.add_argument("-sample-len", "-sl",
                        type=int, default=200_000,
                        help="Specifies the length (in nr of samples) of the sample "
                             "with which the full data is auto correlated with in. Default is 200_000. more = slower")
    parser.add_argument("-correlation-data-interval", "-cdi",
                        type=str, default="20:-200",
                        help="Specifies the interval of the correlation_data that is regarded "
                             "in the form start_sample:end_sample. Default is 20:-200.")
    parser.add_argument("-correlation-threshold", "-cth",
                        type=int,
                        help="Specifies the threshold at which a correlation spike is interpreted as a repeat. "
                             "Default is automatic.")
    parser.add_argument("-max-cluster-len", "-mcl",
                        type=int, default=100_000,
                        help="Specifies the maximum length (in nr. samples) of a cluster "
                             "of which's maximum is interpreted as the repeat time code. Default is 100_000.")
    parser.add_argument("--cpu",
                        help="Forces usage of the cpu correlation data gathering method (weigh slower).",
                        action='store_true')
    parser.add_argument("--force-recalculate", "--fc",
                        help="Forces the recalculation of the correlation data whether or not it's cached.",
                        action='store_true')
    parser.add_argument("--dont-cache", "--dc",
                        help="If activated prohibits caching of expensive calculations.",
                        action='store_true')
    parser.add_argument("--plot-correlation-data", "--pcd",
                        help="Plots the correlation data with matplotlib.",
                        action='store_true')
    parsed_args = parser.parse_args()
    print(parsed_args)
    print(f"* Opening file {parsed_args.input_file!r}")
    in_wav: wave.Wave_read = wave.open(parsed_args.input_file)
    audio = wave_to_np(
        wav=in_wav
    )

    # noinspection PyTypeChecker
    parsed_interval: tuple[int, int] = tuple(int(part) for part in parsed_args.correlation_data_interval.split(":"))

    extracted_audio = extract_audio(
        audio,
        force_recalculate=parsed_args.force_recalculate,
        dont_cache=parsed_args.dont_cache,
        force_cpu=parsed_args.cpu,
        plot_correlation_data=parsed_args.plot_correlation_data,
        correlation_data_interval=parsed_interval,
        sample_start=parsed_args.sample_start,
        sample_len=parsed_args.sample_len,
        correlation_threshold=parsed_args.correlation_threshold,
        max_cluster_len=parsed_args.max_cluster_len
    )

    print("* Saving")
    with wave.open(parsed_args.output_file, "wb") as wav:
        wav.setparams(in_wav.getparams())
        save(extracted_audio, wav)


if __name__ == "__main__":
    main()
