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

import (
	"fmt"
	"log"

	. "github.com/bhojpur/neuron/pkg/engine"
)

// Basic example of representing mathematical equations as graphs.
//
// In this example, we want to represent the following equation
//		z = x + y
func Example_basic() {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(g)
	defer machine.Close()

	// set initial values then run
	Let(x, 3.0)
	Let(y, 4.5)
	if err = machine.RunAll(); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v", z.Value())
	// Output: 7.5
}
