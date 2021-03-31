We want to extract the colors from an image. There are many tools that will do
this, by picking 5 or so colors. But we want to extract ALL the colors. How?
It's not so difficult, as long as we have the farbfeld utilities installed:

    jpegff < input.jpg | xxd -ps -c 8 | tail -n+3 | sort | uniq > output.txt

Now we have a complete listing of every color, in convenient hexadecimal format.
We can view this as an image by reattaching farbfeld headers and passing it back
through xxd. However, the image will be very ugly, since very different colors
are interwoven. A sample is attached:

<img src=mountain.jpg>

So, we want to produce an image that will contain every color from the original
(or near to it), but where the colors smoothly blend between one another. Such
an image would describe a surface, where the 2d coordinate of a point on the
image is mapped to a 3d point in the color space. Such a surface might have very
complex topography, depending on the source image. We can describe such a
surface in the general case using a neural network.

Specifically, we want a neural network that will compress our input colors, each
of which has three dimensions, down to two dimensions. Then a second neural
network will take the two dimensional value, and try and guess which color it
was originally. For training purposes, we can do this using a single neural
network, whose middle layer only has two parameters, and whose input and output
both have 3 parameters. We train it by putting a color as input, and minimizing
the magnitude of the difference between that input and the output. This general
form is called an autoencoder.

The project uses a bash script to convert a jpeg to a list of colors, as above,
gives it to a python script that trains the neural network and produces the
output, then converts it back to a jpeg. 
