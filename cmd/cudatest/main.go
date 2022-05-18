package main

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

// It tests the existence of CUDA by running a simple Go program that uses CUDA.

import (
	"fmt"
	"log"
	"os"

	"github.com/bhojpur/neuron/pkg/drivers/cu"
)

func main() {
	fmt.Printf("\nBhojpur Neuron - CUDA version: %v\n", cu.Version())
	devices, err := cu.NumDevices()
	if err != nil {
		log.Info("issue found: %+v", err)
		os.Exit(1)
	}
	fmt.Printf("CUDA devices: %v\n\n", devices)

	for d := 0; d < devices; d++ {
		name, _ := cu.Device(d).Name()
		cr, _ := cu.Device(d).Attribute(cu.ClockRate)
		mem, _ := cu.Device(d).TotalMem()
		maj, _ := cu.Device(d).Attribute(cu.ComputeCapabilityMajor)
		min, _ := cu.Device(d).Attribute(cu.ComputeCapabilityMinor)
		fmt.Printf("Device %d\n========\nName      :\t%q\n", d, name)
		fmt.Printf("Clock Rate:\t%v kHz\n", cr)
		fmt.Printf("Memory    :\t%v bytes\n", mem)
		fmt.Printf("Compute   : \t%d.%d\n\n", maj, min)
	}
}
