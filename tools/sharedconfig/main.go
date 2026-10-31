// sharedconfig exports Server-owned policy for the repository drift generator.
package main

import (
	"encoding/json"
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"os"
)

func main() {
	if err := json.NewEncoder(os.Stdout).Encode(control.DefaultSharedConfig()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
