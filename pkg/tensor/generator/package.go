package generator

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
	"io"
)

func WritePkgName(f io.Writer, pkg string) {
	switch pkg {
	case TensorPkgLoc:
		fmt.Fprintf(f, "// %s\n\npackage tensor\n\n", Genmsg)
	case NativePkgLoc:
		fmt.Fprintf(f, "// %s\n\npackage native\n\n", Genmsg)
	case ExecLoc:
		fmt.Fprintf(f, "// %s\n\npackage execution\n\n", Genmsg)
	case StorageLoc:
		fmt.Fprintf(f, "// %s\n\npackage storage\n\n", Genmsg)
	default:
		fmt.Fprintf(f, "// %s\n\npackage unknown\n\n", Genmsg)
	}
}

const importUnqualifiedTensor = `import . "github.com/bhojpur/neuron/pkg/tensor"
`
