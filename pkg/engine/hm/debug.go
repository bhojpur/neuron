//go:build debug
// +build debug

package hm

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
	"os"
	"strings"
	"sync/atomic"
)

// DEBUG returns true when it's in debug mode
const DEBUG = false

var tabcount uint32

var _logger_ = log.New(os.Stderr, "", 0)
var replacement = "\n"

func tc() int {
	return int(atomic.LoadUint32(&tabcount))
}

func enterLoggingContext() {
	atomic.AddUint32(&tabcount, 1)
	tabs := tc()
	_logger_.SetPrefix(strings.Repeat("\t", tabs))
	replacement = "\n" + strings.Repeat("\t", tabs)
}

func leaveLoggingContext() {
	tabs := tc()
	tabs--

	if tabs < 0 {
		atomic.StoreUint32(&tabcount, 0)
		tabs = 0
	} else {
		atomic.StoreUint32(&tabcount, uint32(tabs))
	}
	_logger_.SetPrefix(strings.Repeat("\t", tabs))
	replacement = "\n" + strings.Repeat("\t", tabs)
}

func logf(format string, others ...interface{}) {
	if DEBUG {
		// format = strings.Replace(format, "\n", replacement, -1)
		s := fmt.Sprintf(format, others...)
		s = strings.Replace(s, "\n", replacement, -1)
		_logger_.Println(s)
		// _logger_.Printf(format, others...)
	}
}
