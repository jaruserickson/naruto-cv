from pytube import YouTube

# - [Naruto Shippuden Opening 18](https://www.youtube.com/watch?v=HdgD7E6JEE4)
#  - [Naruto Opening 2](https://www.youtube.com/watch?v=SRn99oN1p_c)
#  - [Naruto and Hinata Wedding](https://www.youtube.com/watch?v=BoMBsDIGkKI)

yt = YouTube('https://www.youtube.com/watch?v=SRn99oN1p_c').streams.first().download()
