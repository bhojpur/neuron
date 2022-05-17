package images

// Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"os"
)

func GrayScaleImageFromPath(path string) (*image.Gray, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("couldn't open: %s, %w", path, err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("couldn't decode image %w", err)
	}

	var (
		bounds = img.Bounds()
		gray   = image.NewGray(bounds)
	)
	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}

	return gray, nil
}
func Float64sToImage(data []float64) (image.Image, error) {
	b := make([]byte, len(data))
	for i := 0; i < len(b); i++ {
		b[i] = 128 - byte(data[i]*255.0)
	}

	return BytesToImage(b)
}

func BytesToImage(data []byte) (image.Image, error) {
	bounds := image.Rect(0, 0, 28, 28)
	gray := image.NewGray(bounds)

	fmt.Println(data)

	index := 0
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			gray.SetGray(i, j, color.Gray{Y: data[index]})
			fmt.Printf("%d ", data[index])
			index++
		}
	}
	return gray, nil
}
