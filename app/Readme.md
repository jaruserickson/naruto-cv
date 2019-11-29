# naruto-cv/app

## Notes
This folder contains the main application class and gui for the project.

The structure is layed out as follows:
1. `main.py`
    - This is the main function and is the entry-point to the entire application. Run this in the command line while passing the path to the video you wish to run as one of its arguments
2. `application.py`
    - The application class is at the top of the class hierachy, it holds all the input parameters and serves as a go-between for the algo and the gui.
3. `algoctrl.py`
    - This is where all the algo will run. The algorithm control object separates the processing of data from the higher level operations such as video display.
4. `vidctrl.py`
    - This is where the video is loaded and displayed. The video control object will handle any visualizations that we may wish to feature on top of the original video.
        
The application is multithreaded, that is, the `AlgoCtrl` object and the `VidCtrl` object are run in separate threads and are synchronized such that each waits for the other when necessary. Frames are read inside `VidCtrl` and sent to `AlgoCtrl` where processing takes place. Once the algo is done with the frame, `AlgoCtrl` sends it back to `VidCtrl` for display. 

`VidCtrl` itself is split into two parts - a `VidReader` and a `VidPlayer`. The `VidReader`'s job is to read frames from file and send them to the algo. The `VidPlayer`'s job is to receive processed frames from algo and then display them. These were also made multithreaded so that the reader could provide frames to the algo "on the fly" while the vidplayer is still "playing" the previous frame. Communication between the `VidReader` and the `VidPlayer` is established through the `VidCtrl` object itself, and is needed so the the vidplayer can control the order in which frames are displayed.

Support has also been added for image sets. An image folder can be passed in as argument to the application (and `mode` set to `images`) which will trigger a separate `ImReader` object to take the place of the `VidReader`.

## Usage
To run the application, simply execute
```
python app/main.py --vid-file <path-to-video>
```

Available optional arguments:
- `-h`, `--help`            show help message
- `--vid-file`              video file
- `mode`                    video mode - can be either `video` to read from a video or `images` to read from a folder of images
- `fps`                     video frames per second

These are the current keyboard shortcuts:
- `<q>` quits the application
- `video` mode
    - `<spacebar>` pauses/plays the video (including processing)
    - `<n>` skips forward in the video by 10 seconds
- `images` mode
    - `<p>` go back one frame (only available in `images` mode)
    - `<any key>` go to next frame

## Extras

`video_downloader.py` - Downloads a video from YouTube given a link.
