"""
module containing functions to use ffprobe to parse video frame info
"""
from __future__ import print_function
import subprocess
import cStringIO


class base_frame(object):
    """
    Base Frame from FFProbe
    [FRAME]
    media_type=video
    stream_index=0
    key_frame=0
    pkt_pts=11745667
    pkt_pts_time=130.507411
    pkt_dts=11745667
    pkt_dts_time=130.507411
    best_effort_timestamp=11745667
    best_effort_timestamp_time=130.507411
    pkt_duration=3003
    pkt_duration_time=0.033367
    pkt_pos=86509020
    pkt_size=13294
    ...
    [/FRAME]
    """
    def __init__(self, buf, parser):
        """
        Constructs a base ffprobe frame
        :param buf: buffer containing frame info
        :param parser: ffprobe frame parser
        """
        self.stream_index = parser.get_int(buf)
        self.key_frame = parser.get_int(buf)
        self.pkt_pts = parser.get_int(buf)
        self.pkt_pts_time = parser.get_float(buf)
        self.pkt_dts = parser.get_int(buf)
        self.pkt_dts_time = parser.get_float(buf)
        self.best_effort_timestamp = parser.get_int(buf)
        self.best_effort_timestamp_time = parser.get_float(buf)
        self.pkt_duration = parser.get_int(buf)
        self.pkt_duration_time = parser.get_float(buf)
        self.pkt_pos = parser.get_int(buf)
        self.pkt_size = parser.get_int(buf)


class audio_frame(base_frame):
    """
    Audio Frame from FFProbe
    [FRAME]
    ...
    sample_fmt=s16p
    nb_samples=1152
    channels=2
    channel_layout=stereo
    [/FRAME]
    """
    def __init__(self, buf, parser):
        """
        Constructs an Audio Frame from FFprobe
        :param buf: buffer containing ffprobe frame info
        :param parser: ffprobe frame parser
        """
        super(audio_frame, self).__init__(buf, parser)
        self.media_type = 'audio'
        self.sample_fmt = parser.get_str(buf)
        self.nb_samples = parser.get_int(buf)
        self.channels = parser.get_int(buf)
        self.channel_layout = parser.get_str(buf)


class video_frame(base_frame):
    """
    Video Frame from FFProbe
    [FRAME]
    ...
    width=720
    height=480
    pix_fmt=yuv420p
    sample_aspect_ratio=1:1
    pict_type=B
    coded_picture_number=3889
    display_picture_number=0
    interlaced_frame=0
    top_field_first=0
    repeat_pict=0
    [/FRAME]
    """
    def __init__(self, buf, parser):
        """
        Constructs a Video Frame from ffprobe
        :param buf: buffer containing ffprobe frame info
        :param parser: ffprobe frame parser
        """
        super(video_frame, self).__init__(buf, parser)
        self.media_type = 'video'
        self.width = parser.get_int(buf)
        self.height = parser.get_int(buf)
        self.pix_fmt = parser.get_str(buf)
        self.sample_aspect_ratio = parser.get_str(buf)
        self.pict_type = parser.get_str(buf)
        self.coded_picture_number = parser.get_int(buf)
        self.display_picture_number = parser.get_int(buf)
        self.interlaced_frame = parser.get_int(buf)
        self.top_field_first = parser.get_int(buf)
        self.repeat_pict = parser.get_int(buf)


class side_data(object):
    """
    Side Data from FFProbe
    [SIDE_DATA]
    side_data_type=GOP timecode
    side_data_size=8
    timecode=00:00:00:00
    [/SIDE_DATA]
    """
    def __init__(self, buf, parser):
        """
        Constructs side data frame
        :param buf: buffer containing ffprobe frame info
        :param parser: ffprobe frame parser
        """
        self.side_data_type = parser.get_str(buf)
        self.side_data_size = parser.get_int(buf)
        self.timecode = parser.get_str(buf)


class ffprobe_frame_info_parser(object):
    """
    ffprobe frame parser, reads ffprobe entries and extracts key, value pairs
    """
    def get_str(self, buf, sep='='):
        _, value = buf.readline().split(sep)
        return value[:-1]

    def get_int(self, buf, sep='='):
        _, value = buf.readline().split(sep)
        value = value[:-1]
        if value == 'N/A':
            value = -1
        else:
            value = int(value)
        return value

    def get_float(self, buf, sep='='):
        _, value = buf.readline().split(sep)
        value = value[:-1]
        if value == 'N/A':
            value = float('nan')
        else:
            value = float(value)
        return value

    def get_entry(self, buf, sep='='):
        key, value = buf.readline().split(sep)
        value = value[:-1]
        return key, value


def peek_line(buf):
    pos = buf.tell()
    line = buf.readline()
    buf.seek(pos)
    return line


def ffprobe_video(filename):
    """
    probes a video using ffprobe subprocess
    :param filename: video file to probe
    :return: list of audio, video frames
    """
    command = ["ffprobe", "-show_frames", filename]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    video_frames = []
    audio_frames = []
    p = ffprobe_frame_info_parser()
    buf = cStringIO.StringIO(out)
    while True:
        line = buf.readline()
        if line == '':
            break
        else:
            info_type = line[:-1]
            if info_type == '[FRAME]':
                media_type = p.get_str(buf)
                if media_type == "video":
                    frame = video_frame(buf, p)
                    video_frames.append(frame)
                    # check if [SIDE_DATA] exists
                    line = peek_line(buf)[:-1]
                    if line == '[SIDE_DATA]':
                        _ = buf.readline()  # read the header [SIDE_DATA]
                        _ = side_data(buf, p)
                        buf.readline()  # read the end tag [/SIDE_DATA]
                else:
                    frame = audio_frame(buf, p)
                    audio_frames.append(frame)
                buf.readline()  # read the end tag [/FRAME]
    return audio_frames, video_frames


def main():
    audio_frames, video_frames = ffprobe_video('s01.mpg')
    assert len(video_frames) == 3890


if __name__ == '__main__':
    main()
