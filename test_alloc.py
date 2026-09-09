import llm_allocator_cpp

def main():
    print("=== TESTING C++ BINDINGS FROM PYTHON ===")
    
    allocator = llm_allocator_cpp.PageAllocator(total_blocks=100, block_size=16)
    table = llm_allocator_cpp.SequenceBlockTable(allocator)

    for _ in range(35):
        table.append_token(allocator)

    print(f"Allocated tokens: {table.get_tokens_count()}")
    print(f"Physical DRAM Block IDs: {table.get_physical_blocks()}")
    print(f"Allocator Utilization: {allocator.get_utilization_percentage():.2f}%")

    first_block = table.get_physical_blocks()[0]
    allocator.share_block(first_block)
    print(f"Shared block {first_block} for prefix caching.")

    table.release(allocator)
    print(f"After table release - Utilization: {allocator.get_utilization_percentage():.2f}%")

    print(f"Free blocks: {allocator.get_num_free_blocks()} / {allocator.get_total_blocks()}")

    allocator.free_block(first_block)
    print(f"After sharing release - Utilization: {allocator.get_utilization_percentage():.2f}%")

    print("\n--- Testing Defensive Double-Free Guard ---")
    try:
        allocator.free_block(first_block)
    except RuntimeError as e:
        print(f"SUCCESS: Caught expected C++ exception in Python: {e}")

if __name__ == "__main__":
    main()