FROM golang:1.24 AS builder

WORKDIR /app

COPY go/go.* ./ 
RUN go mod download

COPY go/ . 

RUN go mod tidy

# Esegui la build
RUN go build -o main

FROM ubuntu:24.04

WORKDIR /app

COPY --from=builder /app/main .

RUN chmod +x main

ENTRYPOINT ["/app/main"]
