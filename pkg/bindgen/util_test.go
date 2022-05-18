package bindgen

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

import "testing"

var snaketests = []struct {
	input, output string
	exported      bool
}{
	{"hello_world", "helloWorld", false},
	{"Hello_World", "HelloWorld", true},
	{"Hellow_Sekai_World", "hellowSekaiWorld", false},
	{"Hellow_Sekai_World", "HellowSekaiWorld", true},
	{"Hellow_Sekai_World_123", "HellowSekaiWorld123", true},
	{"Héllow_Sekai_World", "héllowSekaiWorld", false},
	{"_trailing_under||scores_", "TrailingUnder||scores", true}, // this is not a valud function or name, but added for completeness sake
}

func TestSnake2Camel(t *testing.T) {
	for _, st := range snaketests {
		out := Snake2Camel(st.input, st.exported)
		if out != st.output {
			t.Fatalf("Failed on Input %q. Wanted %q. Got %q", st.input, st.output, out)
		}
	}
}
