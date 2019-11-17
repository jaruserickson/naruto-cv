"""
video downloader
"""

import pafy


def download_vid(url='https://www.youtube.com/watch?v=HdgD7E6JEE4'):
    v = pafy.new(url)
    stream = None
    
    for s in v.videostreams:
        if s.quality == '1280x720':
            stream = s
            break
        
    if stream:
        print(f"Downloading '{stream.title}'")
        stream.download()
    
    
    return 0


if __name__ == '__main__':
    download_vid()
