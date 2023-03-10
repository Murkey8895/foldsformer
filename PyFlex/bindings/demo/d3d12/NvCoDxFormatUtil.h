#ifndef NV_CO_DX_FORMAT_UTIL_H
#define NV_CO_DX_FORMAT_UTIL_H

#define NOMINMAX
#include <dxgi.h>

namespace nvidia { 
namespace Common {

struct DxFormatUtil
{
	enum UsageType
	{
		USAGE_UNKNOWN,						///< Generally used to mark an error
		USAGE_TARGET,						///< Format should be used when written as target
		USAGE_DEPTH_STENCIL,				///< Format should be used when written as depth stencil	
		USAGE_SRV,							///< Format if being read as srv
		USAGE_COUNT_OF,
	};
	enum UsageFlag
	{
		USAGE_FLAG_MULTI_SAMPLE = 0x1,
		USAGE_FLAG_SRV = 0x2,
	};

		/// Given the usage, flags, and format will return the most suitable format. Will return DXGI_UNKNOWN if combination is not possible
	static DXGI_FORMAT calcFormat(UsageType usage, DXGI_FORMAT format);
		/// Calculate appropriate format for creating a buffer for usage and flags
	static DXGI_FORMAT calcResourceFormat(UsageType usage, int usageFlags, DXGI_FORMAT format);
		/// True if the type is 'typeless'
	static bool isTypeless(DXGI_FORMAT format);

		/// Returns number of bits used for color channel for format (for channels with multiple sizes, returns smallest ie RGB565 -> 5)
	static int getNumColorChannelBits(DXGI_FORMAT fmt);

};

} // namespace Common 
} // namespace nvidia 

#endif // NV_CO_DX12_RESOURCE_H
