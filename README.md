# AutoCorrelationAudioExtraction
extract repeating audio that is overlayed with a noise

# Usage
`auto_correlation_audio_extraction.py` is a command-line program. 

You can view all options with \
`python auto_correlation_audio_extraction.py -h`
 
# Example
In this **very** ideal situation, this is the input:

https://user-images.githubusercontent.com/37810842/136675428-e89afdd3-95d8-4dcc-89a4-cd85a23fc68a.mp4

I overlayed a section of the song Jealous Girl by Lana Del Rey with Lorem Ipsum read by a tts.
Notice how the section of the song repeats 9 times but the spoken text doesn't.

This is the plotted correlation data:
![image](https://user-images.githubusercontent.com/37810842/136675517-41dc2afd-4f0e-4d58-8981-c46390426329.png)
You can see the spikes showing a repeat.

This is the output

https://user-images.githubusercontent.com/37810842/136675439-276888cd-0594-431c-bc8b-6c953cf62239.mp4

You can hear that the noise is quiter relative to the song section.
