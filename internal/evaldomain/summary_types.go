package evaldomain

type Counts struct {
	Expected           int `json:"expected"`
	Eligible           int `json:"eligible"`
	Unsupported        int `json:"unsupported"`
	Blocked            int `json:"blocked"`
	Submitted          int `json:"submitted"`
	Terminal           int `json:"terminal"`
	Missing            int `json:"missing"`
	Conflicting        int `json:"conflicting"`
	CollectionComplete int `json:"collectionComplete"`
	Scored             int `json:"scored"`
	QualityPassed      int `json:"qualityPassed"`
	ExecutionSucceeded int `json:"executionSucceeded"`
	EndToEndPassed     int `json:"endToEndPassed"`
}

type Ratio struct {
	Numerator   int      `json:"numerator"`
	Denominator int      `json:"denominator"`
	Value       *float64 `json:"value"`
}

type Quality struct {
	ExecutionSuccess   Ratio `json:"executionSuccess"`
	EndToEndPass       Ratio `json:"endToEndPass"`
	ConditionalQuality Ratio `json:"conditionalQuality"`
}

type Summary struct {
	Counts               map[string]Counts  `json:"counts"`
	Quality              map[string]Quality `json:"quality"`
	TerminalPairs        int                `json:"terminalPairs"`
	CompleteQualityPairs int                `json:"completeQualityPairs"`
	CompleteTokenPairs   int                `json:"completeTokenPairs"`
	Conclusion           string             `json:"conclusion"`
}
