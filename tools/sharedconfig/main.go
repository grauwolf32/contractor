// sharedconfig exports Server-owned policy for the repository drift generator.
package main

import (
	"encoding/json"
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"os"
)

func main() {
	config, err := control.DefaultSharedConfig()
	if err == nil {
		err = json.NewEncoder(os.Stdout).Encode(config)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
