package tensor

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
	"log"
	"math"
	"math/rand"
	"reflect"
	"sort"

	"github.com/bhojpur/neuron/pkg/math32"
)

// SortIndex is similar to numpy's argsort
// TODO: tidy this up
func SortIndex(in interface{}) (out []int) {
	switch list := in.(type) {
	case []int:
		orig := make([]int, len(list))
		out = make([]int, len(list))
		copy(orig, list)
		sort.Ints(list)
		for i, s := range list {
			for j, o := range orig {
				if o == s {
					out[i] = j
					break
				}
			}
		}
	case []float64:
		orig := make([]float64, len(list))
		out = make([]int, len(list))
		copy(orig, list)
		sort.Float64s(list)

		for i, s := range list {
			for j, o := range orig {
				if o == s {
					out[i] = j
					break
				}
			}
		}
	case sort.Interface:
		sort.Sort(list)

		log.Printf("TODO: SortIndex for sort.Interface not yet done.")
	}

	return
}

// SampleIndex samples a slice or a Tensor.
// TODO: tidy this up.
func SampleIndex(in interface{}) int {
	// var l int
	switch list := in.(type) {
	case []int:
		var sum, i int
		// l = len(list)
		r := rand.Int()
		for {
			sum += list[i]
			if sum > r && i > 0 {
				return i
			}
			i++
		}
	case []float64:
		var sum float64
		var i int
		// l = len(list)
		r := rand.Float64()
		for {
			sum += list[i]
			if sum > r && i > 0 {
				return i
			}
			i++
		}
	case *Dense:
		var i int
		switch list.t.Kind() {
		case reflect.Float64:
			var sum float64
			r := rand.Float64()
			data := list.Float64s()
			// l = len(data)
			for {
				datum := data[i]
				if math.IsNaN(datum) || math.IsInf(datum, 0) {
					return i
				}

				sum += datum
				if sum > r && i > 0 {
					return i
				}
				i++
			}
		case reflect.Float32:
			var sum float32
			r := rand.Float32()
			data := list.Float32s()
			// l = len(data)
			for {
				datum := data[i]
				if math32.IsNaN(datum) || math32.IsInf(datum, 0) {
					return i
				}

				sum += datum
				if sum > r && i > 0 {
					return i
				}
				i++
			}
		default:
			panic("not yet implemented")
		}
	default:
		panic("Not yet implemented")
	}
	return -1
}
