package image_quality

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
	"os"
)

func ExampleJPEG() {
	filename := "testdata/smile.jpeg"
	f, err := os.Open(filename)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}

	t, err := JPEG(f)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}

	fmt.Printf("%v %v", t.Shape(), t.Dtype)

	// Output:
	// (12, 14, 4) float64
}

func ExamplePNG() {
	filename := "testdata/smile.png"
	f, err := os.Open(filename)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}

	t, err := PNG(f)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}

	fmt.Printf("%v %v", t.Shape(), t.Dtype())

	// Output:
	// (12, 14, 4) float64
}
