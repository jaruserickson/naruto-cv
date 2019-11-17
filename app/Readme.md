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
        
The application is multithreaded, that is, the `AlgoCtrl` object and the `VidCtrl` object are run in separate threads and are synchronized such that each waits for the other when necessary. `AlgoCtrl` waits for "new frame events" which take place when `VidCtrl` loads a new frame from file. `VidCtrl` waits for 'end frame events' which are signaled when `AlgoCtrl` finishes processing a given frame (this leaves us with the ability to skip frames for processing if later we decide that our algorithm is too slow to run every frame).

## Usage
To run the application, simply execute
```
python app/main.py --vid-file <path-to-video>
```

These are the current keyboard shortcuts:
- `q` quits the application
- `<spacebar>` pauses/plays the video (including processing)
- `n` skips forward in the video by 10 seconds

## Extras

`video_downloader.py` - Downloads a video from YouTube given a link.
