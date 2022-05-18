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

import (
	"bytes"
	"fmt"
	"strings"

	"modernc.org/cc"
)

func ExampleExplore() {
	functions := func(decl *cc.Declarator) bool {
		if !strings.HasPrefix(NameOf(decl), "func") {
			return false
		}
		if decl.Type.Kind() == cc.Function {
			return true
		}
		return false
	}
	enums := func(decl *cc.Declarator) bool {
		if decl.Type.Kind() == cc.Enum {
			return true
		}
		return false
	}
	others := func(decl *cc.Declarator) bool {
		if decl.Type.Kind() == cc.Ptr || decl.Type.Kind() == cc.Struct {
			return true
		}
		return false
	}
	tu, err := Parse(Model(), "testdata/dummy.h")
	if err != nil {
		panic(err)
	}

	if err = Explore(tu, functions, enums, others); err != nil {
		panic(err)
	}

	// Output:
	// func1i
	// func1f
	// func1fp
	// func2i
	// func2f
	// funcErr
	// funcCtx
	//
	// error
	// fntype_t
	//
	// dummy_t
	// dummy2_t
}

func ExampleGenIgnored() {
	functions := func(decl *cc.Declarator) bool {
		if !strings.HasPrefix(NameOf(decl), "func") {
			return false
		}
		if decl.Type.Kind() == cc.Function {
			return true
		}
		return false
	}
	tu, err := Parse(Model(), "testdata/dummy.h")
	if err != nil {
		panic(err)
	}

	var buf bytes.Buffer
	if err := GenIgnored(&buf, tu, functions); err != nil {
		panic(err)
	}
	fmt.Println(buf.String())
	// Output:
	// var ignored = map[string]struct{}{
	// "func1i":{},
	// "func1f":{},
	// "func1fp":{},
	// "func2i":{},
	// "func2f":{},
	// "funcErr":{},
	// "funcCtx":{},
	// }
}

func ExampleGenNameMap() {
	functions := func(decl *cc.Declarator) bool {
		if !strings.HasPrefix(NameOf(decl), "func") {
			return false
		}
		if decl.Type.Kind() == cc.Function {
			return true
		}
		return false
	}

	trans := func(a string) string {
		return strings.ToTitle(strings.TrimPrefix(a, "func"))
	}
	tu, err := Parse(Model(), "testdata/dummy.h")
	if err != nil {
		panic(err)
	}

	var buf bytes.Buffer
	if err := GenNameMap(&buf, tu, "m", trans, functions, false); err != nil {
		panic(err)
	}
	fmt.Println(buf.String())

	// Output:
	// var m = map[string]string{
	// "func1i": "1I",
	// "func1f": "1F",
	// "func1fp": "1FP",
	// "func2i": "2I",
	// "func2f": "2F",
	// "funcErr": "ERR",
	// "funcCtx": "CTX",
	// }

}

func ExampleGenNameMap_2() {
	functions := func(decl *cc.Declarator) bool {
		if !strings.HasPrefix(NameOf(decl), "func") {
			return false
		}
		if decl.Type.Kind() == cc.Function {
			return true
		}
		return false
	}

	trans := func(a string) string {
		return strings.ToTitle(strings.TrimPrefix(a, "func"))
	}
	tu, err := Parse(Model(), "testdata/dummy.h")
	if err != nil {
		panic(err)
	}

	var buf bytes.Buffer
	fmt.Fprintln(&buf, "func init() {")
	if err := GenNameMap(&buf, tu, "m", trans, functions, true); err != nil {
		panic(err)
	}
	fmt.Fprintln(&buf, "}")
	fmt.Println(buf.String())

	// Output:
	// func init() {
	// m = map[string]string{
	// "func1i": "1I",
	// "func1f": "1F",
	// "func1fp": "1FP",
	// "func2i": "2I",
	// "func2f": "2F",
	// "funcErr": "ERR",
	// "funcCtx": "CTX",
	// }
	// }

}

func Example_Advanced() {
	_, err := Parse(Model(), "testdata/adv.h")
	if err != nil {
		panic(err)
	}

	// Output:
	//

}
