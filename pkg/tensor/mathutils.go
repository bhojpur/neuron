//go:build amd64 && !noasm
// +build amd64,!noasm

package tensor

func divmod(a, b int) (q, r int)
