#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

#include <brpc/server.h>
#include <google/protobuf/service.h>
#include <json2pb/pb_to_json.h>
#include <spdlog/spdlog.h>

#include "protocol/gen/transfer_service.pb.h"
#include "transport/base_transport.h"

namespace astate {

using Handler = std::function<ResponseStatus(const std::string&, const void*, size_t)>;

class RpcTransferServiceImpl : public astate::proto::RpcTransferService {
 public:
    RpcTransferServiceImpl() = default;
    ~RpcTransferServiceImpl() override = default;

    void Send(
        ::google::protobuf::RpcController* cntl_base,
        const ::astate::proto::TransferData* request,
        ::astate::proto::StatusReply* response,
        ::google::protobuf::Closure* done) override {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);

        // cntl->set_after_rpc_resp_fn(std::bind(&RpcTransferServiceImpl::CallAfterRpc, std::placeholders::_1,
        //                                       std::placeholders::_2, std::placeholders::_3));
        const std::string& request_name = request->request_name();
        const void* data = request->data().data();
        size_t data_size = request->data().size();

        if (handlers_.find(request_name) == handlers_.end()) {
            response->set_code(404);
            response->set_message("Handler not found");
            SPDLOG_ERROR("RpcTransferServiceImpl: Handler not found for request: {}", request_name);
            return;
        }

        try {
            ResponseStatus status = handlers_[request_name](request_name, data, data_size);
            response->set_code(status.success ? 0 : 1);
            response->set_message(status.status_message);
        } catch (std::exception& e) {
            response->set_code(500);
            response->set_message(e.what());
        }
    }
    static void
    CallAfterRpc(brpc::Controller* cntl, const google::protobuf::Message* req, const google::protobuf::Message* res) {
        // at this time res is already sent to client, but cntl/req/res is not destructed
        std::string req_str;
        std::string res_str;
        json2pb::ProtoMessageToJson(*req, &req_str, NULL);
        json2pb::ProtoMessageToJson(*res, &res_str, NULL);
        SPDLOG_INFO("req:{} res:{}", req_str, res_str);
    }

    void RegisterHandler(const std::string& request_name, Handler handler) {
        handlers_[request_name] = handler;
        SPDLOG_INFO("RpcTransferServiceImpl: Registered handler for request: {}", request_name);
    }

 private:
    std::unordered_map<std::string, Handler> handlers_;
};

} // namespace astate
