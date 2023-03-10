/* Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited. */

#include "NvCoFreeList.h"

#include <stdlib.h> 
#include <string.h>

#define DEFAULT_ALIGNMENT 16

namespace nvidia {
namespace Common {

FreeList::~FreeList()
{
	_deallocateBlocks(m_activeBlocks);
	_deallocateBlocks(m_freeBlocks);
}

void FreeList::_init()
{
	m_top = nullptr;
	m_end = nullptr;
	
	m_activeBlocks = nullptr;
	m_freeBlocks = nullptr;

	m_freeElements = nullptr;

	m_elementSize = 0;
	m_alignment = 1;
	m_blockSize = 0;
	m_blockAllocationSize = 0;
	//m_allocator = nullptr;
}

void FreeList::_init(size_t elementSize, size_t alignment, size_t elemsPerBlock)
{
	//allocator = allocator ? allocator : MemoryAllocator::getInstance();
	//assert(allocator);
	//m_allocator = allocator;

	alignment = (alignment < sizeof(void*)) ? sizeof(void*) : alignment;

	// Alignment must be a power of 2
	assert(((alignment - 1) & alignment) == 0);

	// The elementSize must at least be 
	elementSize = (elementSize >= alignment) ? elementSize : alignment;
	m_blockSize = elementSize * elemsPerBlock;
	m_elementSize = elementSize;
	m_alignment = alignment;

	// Calculate the block size neeed, correcting for alignment 
	const size_t alignedBlockSize = (alignment <= DEFAULT_ALIGNMENT) ? 
		_calcAlignedBlockSize(DEFAULT_ALIGNMENT) : 
		_calcAlignedBlockSize(alignment);

	// Make the block struct size aligned
	m_blockAllocationSize = m_blockSize + alignedBlockSize;

	m_top = nullptr;
	m_end = nullptr;

	m_activeBlocks = nullptr;
	m_freeBlocks = nullptr;			///< Blocks that there are no allocations in

	m_freeElements = nullptr;
}

void FreeList::init(size_t elementSize, size_t alignment, size_t elemsPerBlock)
{
	_deallocateBlocks(m_activeBlocks);
	_deallocateBlocks(m_freeBlocks);
	_init(elementSize, alignment, elemsPerBlock);
}

void FreeList::_deallocateBlocks(Block* block)
{
	while (block)
	{
		Block* next = block->m_next;

#ifdef NV_CO_FREE_LIST_INIT_MEM
		Memory::set(block, 0xfd, m_blockAllocationSize);
#endif

		free(block);
		block = next;
	}
}

bool FreeList::isValidAllocation(const void* dataIn) const
{
	uint8_t* data = (uint8_t*)dataIn;

	Block* block = m_activeBlocks;
	while (block)
	{
		uint8_t* start = block->m_data;
		uint8_t* end = start + m_blockSize;

		if (data >= start && data < end)
		{
			// Check it's aligned correctly
			if ((data - start) % m_elementSize)
			{
				return false;
			}

			// Non allocated data is between top and end
			if (data >= m_top && data < m_end)
			{
				return false;
			}

			// It can't be in the free list
			Element* ele = m_freeElements;
			while (ele)
			{
				if (ele == (Element*)data)
				{
					return false;
				}

				ele = ele->m_next;
			}
			return true;
		}

		block = block->m_next;
	}
	// It's not in an active block -> it cannot be a valid allocation
	return false;
}

void* FreeList::_allocate()
{
	Block* block = m_freeBlocks;
	if (block)
	{
		/// Remove from the free blocks
		m_freeBlocks = block->m_next;
	}
	else
	{
		block = (Block*)malloc(m_blockAllocationSize);
		if (!block)
		{
			// Allocation failed... doh
			return nullptr;
		}
		// Do the alignment
		{
			size_t fix = (size_t(block) + sizeof(Block) + m_alignment - 1) & ~(m_alignment - 1);
			block->m_data = (uint8_t*)fix;
		}
	} 

	// Attach to the active blocks
	block->m_next = m_activeBlocks;
	m_activeBlocks = block;

	// Set up top and end
	m_end = block->m_data + m_blockSize;

	// Return the first element
	uint8_t* element = block->m_data;
	m_top = element + m_elementSize;

	NV_CO_FREE_LIST_INIT_ALLOCATE(element)

	return element;
}

void FreeList::deallocateAll()
{
	Block* block = m_activeBlocks;
	if (block)
	{
		// Find the end block
		while (block->m_next) 
		{
#ifdef NV_CO_FREE_LIST_INIT_MEM
			Memory::set(block->m_data, 0xfd, m_blockSize);	
#endif
			block = block->m_next;
		}
		// Attach to the freeblocks
		block->m_next = m_freeBlocks;
		// The list is now all freelists
		m_freeBlocks = m_activeBlocks;
		// There are no active blocks
		m_activeBlocks = nullptr;
	}

	m_top = nullptr;
	m_end = nullptr;
}

void FreeList::reset()
{
	_deallocateBlocks(m_activeBlocks);
	_deallocateBlocks(m_freeBlocks);

	m_top = nullptr;
	m_end = nullptr;

	m_activeBlocks = nullptr;
	m_freeBlocks = nullptr;			

	m_freeElements = nullptr;
}


void FreeList::_initAllocate(void* mem)
{
	memset(mem, 0xcd, m_elementSize);
}

void FreeList::_initDeallocate(void* mem)
{
	memset(mem, 0xfd, m_elementSize);
}

} // namespace Common 
} // namespace nvidia

