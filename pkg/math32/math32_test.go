//go:build !tinygo
// +build !tinygo

package math32

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
	"os/exec"
	"testing"
)

func TestVetMath32(t *testing.T) {
	linuxArches := []string{"amd64", "arm", "arm64", "s390x", "ppc64le", "riscv64", "386", "mips", "mips64", "mipsle", "mips64le"}

	// Linux architectures
	for _, GOARCH := range linuxArches {
		GOOS := "linux"
		t.Run(fmt.Sprintf("GOOS=%s GOARCH=%s", GOOS, GOARCH), func(t *testing.T) {
			goVet(t, GOOS, GOARCH)
			goBuildVet(t, GOOS, GOARCH)
		})
	}
	// WASM
	t.Run("GOOS=js GOARCH=wasm", func(t *testing.T) {
		goVet(t, "js", "wasm")
		goBuildVet(t, "js", "wasm")
	})

	// AIX
	t.Run("GOOS=aix GOARCH=ppc64", func(t *testing.T) {
		goVet(t, "aix", "ppc64")
		goBuildVet(t, "aix", "ppc64")
	})
}

func goVet(t *testing.T, GOOS, GOARCH string) {
	env := append(os.Environ(), "GOOS="+GOOS, "GOARCH="+GOARCH)
	cmd := exec.Command("go", "vet", ".")
	cmd.Env = env
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Error(string(output), err)
	}
}

func goBuildVet(t *testing.T, GOOS, GOARCH string) {
	env := append(os.Environ(), "GOOS="+GOOS, "GOARCH="+GOARCH)
	const buildname = "math32.test"
	defer os.Remove(buildname)
	cmd := exec.Command("go", "test", "-c", "-o="+buildname)
	cmd.Env = env
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Error(string(output), err)
	}
}
