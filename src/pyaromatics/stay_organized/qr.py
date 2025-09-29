import os
from pyzbar.pyzbar import decode
from PIL import Image
import base64

from urllib.parse import urlparse, parse_qs

from google.protobuf import descriptor_pb2, message_factory

# Google Authenticator migration proto definition
# Based on reverse-engineered schema
from google.protobuf import descriptor_pool
from google.protobuf.message_factory import MessageFactory


def build_protos():
    file_desc = descriptor_pb2.FileDescriptorProto()
    file_desc.name = "migration.proto"
    file_desc.package = "google_auth"

    otp = file_desc.message_type.add()
    otp.name = "OtpParameters"
    f = otp.field.add(); f.name="secret"; f.number=1; f.label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL; f.type=descriptor_pb2.FieldDescriptorProto.TYPE_BYTES
    f = otp.field.add(); f.name="name"; f.number=2; f.label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL; f.type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    f = otp.field.add(); f.name="issuer"; f.number=3; f.label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL; f.type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    f = otp.field.add(); f.name="algorithm"; f.number=4; f.label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL; f.type=descriptor_pb2.FieldDescriptorProto.TYPE_INT32
    f = otp.field.add(); f.name="digits"; f.number=5; f.label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL; f.type=descriptor_pb2.FieldDescriptorProto.TYPE_INT32
    f = otp.field.add(); f.name="type"; f.number=6; f.label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL; f.type=descriptor_pb2.FieldDescriptorProto.TYPE_INT32

    mp = file_desc.message_type.add()
    mp.name = "MigrationPayload"
    f = mp.field.add(); f.name="otp_parameters"; f.number=1; f.label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED; f.type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    f.type_name = ".google_auth.OtpParameters"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_desc)
    factory = message_factory.MessageFactory(pool)
    return factory.GetPrototype(pool.FindMessageTypeByName("google_auth.MigrationPayload"))

def is_valid_base32(s):
    s2 = "".join(s.split()).upper()
    pad = "=" * ((8 - len(s2) % 8) % 8)
    try:
        base64.b32decode(s2 + pad, casefold=True)
        return True
    except Exception:
        return False

def extract_secret_from_otpauth(uri):
    q = parse_qs(urlparse(uri).query)
    sec = q.get("secret")
    if not sec:
        return None
    return sec[0].upper().replace("=", "")

def decode_qr_pic(path):
    proto_cls = build_protos()
    img = Image.open(path)
    results = decode(img)
    if not results:
        print("No se detectó QR en la imagen.")
        return

    for r in results:
        data = r.data.decode("utf-8")
        if data.startswith("otpauth://"):
            secret = extract_secret_from_otpauth(data)
            print("Encontrada URI otpauth://")
            print("URI completa:", data)
            if secret:
                print("Secreto (Base32):", secret)
                print("Valido Base32:", is_valid_base32(secret))
            else:
                print("No se encontró parámetro 'secret' en la URI.")

        elif data.startswith("otpauth-migration://"):
            # parse data=... param
            qs = parse_qs(urlparse(data).query)
            b64 = qs.get("data", [None])[0]
            if not b64:
                print("No se encontró 'data' en otpauth-migration://")
                continue
            raw = base64.b64decode(b64)
            payload = proto_cls()
            payload.ParseFromString(raw)
            for p in payload.otp_parameters:
                name = getattr(p, "name", "")
                issuer = getattr(p, "issuer", "")
                secret_bytes = getattr(p, "secret", b"")
                secret_b32 = base64.b32encode(secret_bytes).decode("utf-8").replace("=", "")
                print("----")
                print("Cuenta:", f"{issuer}:{name}" if issuer else name)
                print("Secreto (Base32):", secret_b32)
                print("Valido Base32:", is_valid_base32(secret_b32))
                # print otpauth URI you can paste to apps that accept it
                uri = f"otpauth://totp/{issuer}:{name}?secret={secret_b32}"
                if issuer:
                    uri += f"&issuer={issuer}"
                print("otpauth URI:", uri)
        else:
            print("Contenido QR no reconocido:", data[:120])

if __name__ == "__main__":
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    qr_path = os.path.join(desktop, "WIN_20250929_19_23_58_Pro.jpg")
    decode_qr_pic(qr_path)